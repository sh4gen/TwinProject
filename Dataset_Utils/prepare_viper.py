import os
import random
from pathlib import Path
from PIL import Image
import shutil

def convert_viper_to_market1501(viper_dir, output_dir, train_ratio=0.5):
    """
    Converts VIPeR dataset to Market1501 format, saving images as JPG.
    
    Args:
        viper_dir: Path to VIPeR dataset (containing cam_a and cam_b folders)
        output_dir: Path where Market1501 format dataset will be created
        train_ratio: Ratio of identities to use for training (default 0.5)
    """
    viper_path = Path(viper_dir)
    output_path = Path(output_dir)
    
    # Verify VIPeR structure
    cam_a_path = viper_path / "cam_a"
    cam_b_path = viper_path / "cam_b"
    
    if not cam_a_path.exists() or not cam_b_path.exists():
        print("ERROR: cam_a or cam_b folder not found!")
        return
    
    # Create output directories
    train_dir = output_path / "bounding_box_train"
    test_dir = output_path / "bounding_box_test"
    query_dir = output_path / "query"
    
    for dir_path in [train_dir, test_dir, query_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories at: {output_path}")
    
    # Get all person IDs from cam_a
    person_ids = []
    for img_file in sorted(cam_a_path.glob("*.bmp")):
        person_id = img_file.stem.split('_')[0]
        person_ids.append(person_id)
    
    # Remove duplicates and sort
    person_ids = sorted(list(set(person_ids)))
    print(f"Found {len(person_ids)} unique person IDs")
    
    # Split into train and test sets
    random.seed(42)  # For reproducibility
    random.shuffle(person_ids)
    
    split_point = int(len(person_ids) * train_ratio)
    train_ids = person_ids[:split_point]
    test_ids = person_ids[split_point:]
    
    print(f"Train set: {len(train_ids)} identities")
    print(f"Test/Query set: {len(test_ids)} identities")
    
    # Process images
    total_converted = 0
    
    # Process training set
    print("\nProcessing training set...")
    for person_id in train_ids:
        numeric_id = int(person_id)
        formatted_id = f"{numeric_id:04d}"
        
        # Process cam_a image
        cam_a_files = list(cam_a_path.glob(f"{person_id}_*.bmp"))
        if cam_a_files:
            src_path = cam_a_files[0]
            # CHANGED: Output filename is now .jpg
            dst_filename = f"{formatted_id}_c1s1_01_00.jpg"
            dst_path = train_dir / dst_filename
            
            try:
                with Image.open(src_path) as img:
                    # CHANGED: Convert to RGB and save as JPEG with high quality
                    img.convert('RGB').save(dst_path, 'JPEG', quality=95)
                total_converted += 1
            except Exception as e:
                print(f"Error converting {src_path}: {e}")
        
        # Process cam_b image
        cam_b_files = list(cam_b_path.glob(f"{person_id}_*.bmp"))
        if cam_b_files:
            src_path = cam_b_files[0]
            # CHANGED: Output filename is now .jpg
            dst_filename = f"{formatted_id}_c2s1_01_00.jpg"
            dst_path = train_dir / dst_filename
            
            try:
                with Image.open(src_path) as img:
                    # CHANGED: Convert to RGB and save as JPEG with high quality
                    img.convert('RGB').save(dst_path, 'JPEG', quality=95)
                total_converted += 1
            except Exception as e:
                print(f"Error converting {src_path}: {e}")
    
    # Process test/query set
    print("\nProcessing test/query set...")
    for person_id in test_ids:
        numeric_id = int(person_id)
        formatted_id = f"{numeric_id:04d}"
        
        # cam_a image goes to query
        cam_a_files = list(cam_a_path.glob(f"{person_id}_*.bmp"))
        if cam_a_files:
            src_path = cam_a_files[0]
            # CHANGED: Output filename is now .jpg
            dst_filename = f"{formatted_id}_c1s1_01_00.jpg"
            dst_path = query_dir / dst_filename
            
            try:
                with Image.open(src_path) as img:
                    # CHANGED: Convert to RGB and save as JPEG with high quality
                    img.convert('RGB').save(dst_path, 'JPEG', quality=95)
                total_converted += 1
            except Exception as e:
                print(f"Error converting {src_path}: {e}")
        
        # cam_b image goes to gallery (bounding_box_test)
        cam_b_files = list(cam_b_path.glob(f"{person_id}_*.bmp"))
        if cam_b_files:
            src_path = cam_b_files[0]
            # CHANGED: Output filename is now .jpg
            dst_filename = f"{formatted_id}_c2s1_01_00.jpg"
            dst_path = test_dir / dst_filename
            
            try:
                with Image.open(src_path) as img:
                    # CHANGED: Convert to RGB and save as JPEG with high quality
                    img.convert('RGB').save(dst_path, 'JPEG', quality=95)
                total_converted += 1
            except Exception as e:
                print(f"Error converting {src_path}: {e}")
    
    print(f"\nConversion complete!")
    print(f"Total images converted: {total_converted}")
    print(f"\nDataset statistics:")
    # CHANGED: Count .jpg files
    print(f"- Training images: {len(list(train_dir.glob('*.jpg')))}")
    print(f"- Gallery images: {len(list(test_dir.glob('*.jpg')))}")
    print(f"- Query images: {len(list(query_dir.glob('*.jpg')))}")
    
    # Verify the conversion
    print("\nSample filenames:")
    for split_name, split_dir in [("Train", train_dir), ("Test", test_dir), ("Query", query_dir)]:
        # CHANGED: Look for .jpg files
        files = sorted(list(split_dir.glob('*.jpg')))[:3]
        if files:
            print(f"\n{split_name}:")
            for f in files:
                print(f"  {f.name}")

if __name__ == "__main__":
    # CONFIGURE YOUR PATHS HERE
    VIPER_DATASET_PATH = "/home/ika/Downloads/Datasets/VIPeR/viper"
    OUTPUT_PATH = "/home/ika/yzlm/TwinProject/ReID_Experiments/Viper/data"
    
    # You can adjust the train/test split ratio (e.g., 0.8 for 80% train)
    TRAIN_RATIO = 0.8
    
    # --- IMPORTANT: Clean up the old directory before running ---
    if os.path.exists(OUTPUT_PATH):
        print(f"Removing old directory: {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)
    
    # Run the conversion
    convert_viper_to_market1501(VIPER_DATASET_PATH, OUTPUT_PATH, TRAIN_RATIO)