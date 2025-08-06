import os
import numpy as np
from PIL import Image
import glob
from collections import defaultdict
import json
from tqdm import tqdm
import re

def analyze_reid_dataset(dataset_path, save_stats=True):
    stats = {
        'pixel_mean': [0, 0, 0],
        'pixel_std': [0, 0, 0],
        'num_classes': 0,
        'num_cameras': 0,
        'image_stats': {
            'min_width': float('inf'),
            'max_width': 0,
            'min_height': float('inf'),
            'max_height': 0,
            'avg_width': 0,
            'avg_height': 0,
            'total_images': 0
        },
        'split_stats': {},
        'class_distribution': {},
        'camera_distribution': {},
        'images_per_class': {},
        'dataset_info': {}
    }
    
    # Define splits to analyze
    splits = {
        'train': ['bounding_box_train', 'train'],
        'val': ['bounding_box_val', 'val'],
        'test': ['bounding_box_test', 'test', 'gallery'],
        'query': ['query']
    }
    
    # Regular expression to parse Market1501 format filenames
    # Format: XXXX_cXsX_XXXXXX_XX.jpg
    pattern = re.compile(r'(\d+)_c(\d+)')
    
    print("Analyzing dataset structure...")
    
    # Find which directories exist
    existing_dirs = {}
    for split_name, possible_dirs in splits.items():
        for dir_name in possible_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if os.path.exists(dir_path):
                existing_dirs[split_name] = dir_path
                break
    
    if not existing_dirs:
        print(f"Error: No valid directories found in {dataset_path}")
        return None
    
    print(f"Found directories: {list(existing_dirs.keys())}")
    
    # Collect all images for pixel statistics
    all_images = []
    all_person_ids = set()
    all_camera_ids = set()
    
    # Analyze each split
    for split_name, split_path in existing_dirs.items():
        print(f"\nAnalyzing {split_name} split...")
        
        # Get all images in this split
        image_files = glob.glob(os.path.join(split_path, '*.jpg'))
        image_files.extend(glob.glob(os.path.join(split_path, '*.png')))
        
        stats['split_stats'][split_name] = {
            'num_images': len(image_files),
            'num_ids': 0,
            'num_cameras': 0
        }
        
        split_person_ids = set()
        split_camera_ids = set()
        
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            # Parse filename
            filename = os.path.basename(img_path)
            match = pattern.search(filename)
            
            if match:
                person_id = int(match.group(1))
                camera_id = int(match.group(2))
                
                all_person_ids.add(person_id)
                split_person_ids.add(person_id)
                all_camera_ids.add(camera_id)
                split_camera_ids.add(camera_id)
                
                # Update distributions
                stats['class_distribution'][person_id] = stats['class_distribution'].get(person_id, 0) + 1
                stats['camera_distribution'][camera_id] = stats['camera_distribution'].get(camera_id, 0) + 1
                
                if person_id not in stats['images_per_class']:
                    stats['images_per_class'][person_id] = 0
                stats['images_per_class'][person_id] += 1
            
            # Add to all images list (sample for pixel stats)
            if split_name == 'train' and len(all_images) < 10000:  # Sample up to 10k images
                all_images.append(img_path)
            
            # Get image dimensions
            try:
                img = Image.open(img_path)
                width, height = img.size
                
                stats['image_stats']['min_width'] = min(stats['image_stats']['min_width'], width)
                stats['image_stats']['max_width'] = max(stats['image_stats']['max_width'], width)
                stats['image_stats']['min_height'] = min(stats['image_stats']['min_height'], height)
                stats['image_stats']['max_height'] = max(stats['image_stats']['max_height'], height)
                stats['image_stats']['total_images'] += 1
                
                img.close()
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
        
        stats['split_stats'][split_name]['num_ids'] = len(split_person_ids)
        stats['split_stats'][split_name]['num_cameras'] = len(split_camera_ids)
    
    # Calculate overall statistics
    stats['num_classes'] = len(all_person_ids)
    stats['num_cameras'] = len(all_camera_ids)
    
    # Calculate pixel statistics from training images
    if all_images:
        print(f"\nCalculating pixel statistics from {len(all_images)} images...")
        
        # Initialize arrays for mean and std calculation
        pixel_sum = np.zeros(3)
        pixel_sq_sum = np.zeros(3)
        num_pixels = 0
        
        for img_path in tqdm(all_images, desc="Computing pixel stats"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0
                
                # Accumulate statistics
                pixel_sum += img_array.sum(axis=(0, 1))
                pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
                num_pixels += img_array.shape[0] * img_array.shape[1]
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Calculate mean and std
        pixel_mean = pixel_sum / num_pixels
        pixel_std = np.sqrt(pixel_sq_sum / num_pixels - pixel_mean ** 2)
        
        stats['pixel_mean'] = pixel_mean.tolist()
        stats['pixel_std'] = pixel_std.tolist()
    
    # Calculate average dimensions
    if stats['image_stats']['total_images'] > 0:
        total_width = 0
        total_height = 0
        sample_count = min(1000, stats['image_stats']['total_images'])
        
        print(f"\nCalculating average dimensions from {sample_count} images...")
        
        # Sample images from training set
        if 'train' in existing_dirs:
            sample_images = glob.glob(os.path.join(existing_dirs['train'], '*.jpg'))[:sample_count]
            
            for img_path in sample_images:
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                    total_width += width
                    total_height += height
                    img.close()
                except:
                    pass
            
            stats['image_stats']['avg_width'] = total_width // sample_count
            stats['image_stats']['avg_height'] = total_height // sample_count
    
    # Add dataset summary
    stats['dataset_info'] = {
        'total_images': sum(s['num_images'] for s in stats['split_stats'].values()),
        'unique_identities': stats['num_classes'],
        'unique_cameras': stats['num_cameras'],
        'avg_images_per_identity': sum(stats['images_per_class'].values()) / len(stats['images_per_class']) if stats['images_per_class'] else 0,
        'min_images_per_identity': min(stats['images_per_class'].values()) if stats['images_per_class'] else 0,
        'max_images_per_identity': max(stats['images_per_class'].values()) if stats['images_per_class'] else 0
    }
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nDataset Path: {dataset_path}")
    print(f"Total Images: {stats['dataset_info']['total_images']}")
    print(f"Number of Identities (num_classes): {stats['num_classes']}")
    print(f"Number of Cameras: {stats['num_cameras']}")
    
    print(f"\nPixel Statistics (normalized 0-1):")
    print(f"  Mean (RGB): {[f'{x:.4f}' for x in stats['pixel_mean']]}")
    print(f"  Std (RGB): {[f'{x:.4f}' for x in stats['pixel_std']]}")
    
    print(f"\nImage Dimensions:")
    print(f"  Min: {stats['image_stats']['min_width']}x{stats['image_stats']['min_height']}")
    print(f"  Max: {stats['image_stats']['max_width']}x{stats['image_stats']['max_height']}")
    print(f"  Avg: {stats['image_stats']['avg_width']}x{stats['image_stats']['avg_height']}")
    
    print(f"\nSplit Statistics:")
    for split_name, split_stats in stats['split_stats'].items():
        print(f"  {split_name}: {split_stats['num_images']} images, "
              f"{split_stats['num_ids']} IDs, {split_stats['num_cameras']} cameras")
    
    print(f"\nImages per Identity:")
    print(f"  Average: {stats['dataset_info']['avg_images_per_identity']:.1f}")
    print(f"  Min: {stats['dataset_info']['min_images_per_identity']}")
    print(f"  Max: {stats['dataset_info']['max_images_per_identity']}")
    
    # Generate YAML configuration
    print("\n" + "="*50)
    print("SUGGESTED YAML CONFIGURATION")
    print("="*50)
    
    yaml_config = f"""
dataset:
  num_classes: {stats['num_classes']}

  pixel_mean: {[round(x, 3) for x in stats['pixel_mean']]}
  pixel_std: {[round(x, 3) for x in stats['pixel_std']]}


model:
  input_width: {stats['image_stats']['avg_width'] if stats['image_stats']['avg_width'] > 0 else 128}
  input_height: {stats['image_stats']['avg_height'] if stats['image_stats']['avg_height'] > 0 else 256}
"""
    
    print(yaml_config)
    
    # Save statistics to JSON
    if save_stats:
        output_file = os.path.join(dataset_path, 'dataset_statistics.json')
        
        # Convert sets to lists for JSON serialization
        stats_json = stats.copy()
        stats_json['class_distribution'] = {str(k): v for k, v in stats['class_distribution'].items()}
        stats_json['camera_distribution'] = {str(k): v for k, v in stats['camera_distribution'].items()}
        stats_json['images_per_class'] = {str(k): v for k, v in stats['images_per_class'].items()}
        
        with open(output_file, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"\nStatistics saved to: {output_file}")
    
    return stats


# Example usage
if __name__ == "__main__":
    dataset_path = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTTC+PRCC+ULIRI/data"
    
    stats = analyze_reid_dataset(dataset_path, save_stats=True)