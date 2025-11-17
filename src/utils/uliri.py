import os
import shutil
from pathlib import Path
import re
from collections import defaultdict
import argparse

class ULIRIToMarket1501Converter:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.person_id_mapping = {}
        self.frame_counter = defaultdict(lambda: defaultdict(int))  # cam -> seq -> frame_count
        
    def parse_uliri_filename(self, filename):
        """
        Parse ULI-RI filename: idt033_cam00_rotz03_illu0007.png
        Returns: (person_id, camera_id, rotation, illumination)
        """
        pattern = r'idt(\d+)_cam(\d+)_rotz(\d+)_illu(\d+)\.png'
        match = re.match(pattern, filename)
        if match:
            person_id = match.group(1)
            camera_id = int(match.group(2))
            rotation = int(match.group(3))
            illumination = int(match.group(4))
            return person_id, camera_id, rotation, illumination
        return None, None, None, None
    
    def create_market1501_filename(self, person_id, camera_id, rotation, illumination, bbox_id=0):
        """
        Create Market-1501 style filename: 0001_c1s1_001051_00.jpg
        Format: {person_id}_c{camera_id}s{sequence_id}_{frame_number}_{bbox_id}.jpg
        
        For ULI-RI conversion:
        - Use rotation as sequence identifier (since ULI-RI has different rotations)
        - Use illumination as frame number (scaled appropriately)
        - bbox_id is always 00 since we have one bbox per image
        """
        # Ensure person_id is 4 digits
        formatted_person_id = f"{int(person_id):04d}"
        
        # Camera ID starts from 1 in Market-1501
        formatted_camera_id = camera_id + 1
        
        # Use rotation as sequence ID
        sequence_id = rotation + 1
        
        # Use illumination as frame number (scaled to look realistic)
        frame_number = illumination * 1000 + 1000  # Scale to get numbers like 001000-008000
        
        # Format: person_id_c{cam}s{seq}_{frame}_{bbox}.jpg
        filename = f"{formatted_person_id}_c{formatted_camera_id}s{sequence_id}_{frame_number:06d}_{bbox_id:02d}.jpg"
        return filename
    
    def convert_dataset(self):
        """
        Convert ULI-RI dataset to Market-1501 format with proper query-gallery overlap
        """
        # Create output directories
        bounding_box_train_dir = self.output_dir / "bounding_box_train"
        bounding_box_test_dir = self.output_dir / "bounding_box_test"
        query_dir = self.output_dir / "query"
        gt_bbox_dir = self.output_dir / "gt_bbox"
        
        # Create directories
        for dir_path in [bounding_box_train_dir, bounding_box_test_dir, query_dir, gt_bbox_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all person directories
        person_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        person_dirs.sort()
        
        # Split persons: 70% train, 30% test+query (shared identities)
        total_persons = len(person_dirs)
        train_split = int(0.7 * total_persons)
        
        train_persons = person_dirs[:train_split]
        test_query_persons = person_dirs[train_split:]  # Shared identities for test and query
        
        print(f"Total persons: {total_persons}")
        print(f"Train persons: {len(train_persons)}")
        print(f"Test+Query persons (shared identities): {len(test_query_persons)}")
        
        # Process training set
        print("\nProcessing training set...")
        self._process_person_set(train_persons, bounding_box_train_dir, "train")
        
        # Process test and query sets with shared identities but different images
        print("\nProcessing test and query sets with shared identities...")
        self._process_test_query_sets(test_query_persons, bounding_box_test_dir, query_dir)
        
        # Create gt_bbox (same as bounding_box_test for evaluation)
        print("\nCreating ground truth bbox...")
        for file in bounding_box_test_dir.glob("*.jpg"):
            shutil.copy2(file, gt_bbox_dir / file.name)
        
        print(f"\nConversion completed!")
        print(f"Output directory: {self.output_dir}")
        
    def _process_test_query_sets(self, person_dirs, test_output_dir, query_output_dir):
        """
        Process test and query sets ensuring shared identities but different images
        For each person, put some images in test (gallery) and others in query
        """
        for person_dir in person_dirs:
            # Get all images for this person
            image_files = list(person_dir.glob("*.png"))
            
            if len(image_files) < 2:
                print(f"Warning: Person {person_dir.name} has less than 2 images, skipping...")
                continue
            
            # Sort images to ensure consistent splitting
            image_files.sort()
            
            # Split images: 70% to test (gallery), 30% to query
            # Ensure at least 1 image in each set
            query_count = max(1, int(0.3 * len(image_files)))
            test_count = len(image_files) - query_count
            
            query_images = image_files[:query_count]
            test_images = image_files[query_count:]
            
            print(f"Person {person_dir.name}: {len(test_images)} test images, {len(query_images)} query images")
            
            # Process test images
            for img_file in test_images:
                self._convert_and_copy_image(img_file, test_output_dir, "test")
            
            # Process query images  
            for img_file in query_images:
                self._convert_and_copy_image(img_file, query_output_dir, "query")
    
    def _convert_and_copy_image(self, img_file, output_dir, set_type):
        """Convert and copy a single image file"""
        # Parse original filename
        person_id, camera_id, rotation, illumination = self.parse_uliri_filename(img_file.name)
        
        if person_id is None:
            print(f"Warning: Could not parse filename {img_file.name}")
            return
        
        # Create Market-1501 style filename
        new_filename = self.create_market1501_filename(
            person_id, camera_id, rotation, illumination
        )
        
        # Copy and rename file
        output_path = output_dir / new_filename
        
        try:
            shutil.copy2(img_file, output_path)
            print(f"Converted ({set_type}): {img_file.name} -> {new_filename}")
        except Exception as e:
            print(f"Error copying {img_file.name}: {e}")
    
    def _process_person_set(self, person_dirs, output_dir, set_type):
        """Process a set of persons (train only - test/query handled separately)"""
        for person_dir in person_dirs:
            # Get all images in the person directory
            image_files = list(person_dir.glob("*.png"))
            
            for img_file in image_files:
                self._convert_and_copy_image(img_file, output_dir, set_type)
    
    def generate_info_file(self):
        """Generate dataset information file"""
        info_content = """Dataset Conversion Information
==============================

Original Dataset: ULI-RI
Target Format: Market-1501

Conversion Details:
- ULI-RI format: idt{person_id}_cam{camera_id}_rotz{rotation}_illu{illumination}.png
- Market-1501 format: {person_id}_c{camera_id}s{sequence_id}_{frame_number}_{bbox_id}.jpg

Mapping Rules:
- Person ID: Extracted from 'idt' prefix, zero-padded to 4 digits
- Camera ID: Original camera_id + 1 (Market-1501 cameras start from 1)
- Sequence ID: rotation + 1 (using rotation as sequence identifier)
- Frame Number: illumination * 1000 + 1000 (scaled for realistic frame numbers)
- Bbox ID: Always 00 (single bounding box per image)

Dataset Split:
- Training: 70% of persons (disjoint identities)
- Test & Query: 30% of persons (shared identities, different images)
  - For each test identity: ~70% images in gallery, ~30% images in query
- Ground Truth: Copy of test set

Directory Structure:
- bounding_box_train/: Training images (disjoint person IDs)
- bounding_box_test/: Test gallery images (shared person IDs with query)
- query/: Query images for evaluation (shared person IDs with test)
- gt_bbox/: Ground truth bounding boxes (copy of test set)

Important Notes:
- Query and gallery (test) sets share the same person identities
- Each person has different images in query vs gallery to enable proper evaluation
- This ensures all query identities appear in the gallery for valid ReID evaluation
"""
        
        info_file = self.output_dir / "conversion_info.txt"
        with open(info_file, 'w') as f:
            f.write(info_content)
        
        print(f"Dataset information saved to: {info_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert ULI-RI dataset to Market-1501 format')
    parser.add_argument('--source', '-s', required=True, help='Source ULI-RI dataset directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory for Market-1501 format')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist!")
        return
    
    # Create converter and run conversion
    converter = ULIRIToMarket1501Converter(args.source, args.output)
    converter.convert_dataset()
    converter.generate_info_file()
    
    print("\n" + "="*50)
    print("CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    # Example usage when running directly
    # Uncomment and modify paths as needed:
    
    source_dir = "/home/ika/Downloads/Datasets/ULIRI"
    output_dir = "/home/ika/Downloads/Datasets/ULIRI_Formatted"
    # 
    converter = ULIRIToMarket1501Converter(source_dir, output_dir)
    converter.convert_dataset()
    converter.generate_info_file()
    
    #main()