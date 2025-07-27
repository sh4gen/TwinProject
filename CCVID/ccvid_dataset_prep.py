import os
from shutil import copyfile
import glob

# Dataset paths - modify these as needed
download_path = 'dataset'  # Base path for CCVID dataset
download_path2 = '/home/ika/yzlm/TwinProject/CCVID/dataset'  # Alternative path if dataset is elsewhere

# Handle dataset location
if not os.path.isdir(download_path):
    if os.path.isdir(download_path2):
        os.system('mv {} {}'.format(download_path2, download_path))
    else:
        print('Please change the download_path to your CCVID dataset location')
        exit()

# Create market1501-style directory structure
save_path = os.path.join(download_path, 'market1501_format')
os.makedirs(save_path, exist_ok=True)

# Camera ID mapping (you may need to adjust based on your dataset)
# For CCVID, we'll use session numbers as camera IDs
camera_mapping = {
    'session1': 1,
    'session2': 2, 
    'session3': 3
}

def create_market1501_filename(identity, camera_id, sequence_num, frame_num=0):
    """Create Market1501 style filename: XXXX_cXsX_XXXXXX_XX.jpg"""
    # Ensure identity is 4 digits with leading zeros
    identity_str = str(identity).zfill(4)
    # Create filename: personID_cameraID_sequenceNum_frameNum.jpg
    filename = f"{identity_str}_c{camera_id}s1_{str(sequence_num).zfill(6)}_{str(frame_num).zfill(2)}.jpg"
    return filename

def process_split(split_name, target_dir):
    """Process a dataset split (query/gallery/train) into Market1501 format"""
    txt_file = os.path.join(download_path, f'{split_name}.txt')
    if not os.path.exists(txt_file):
        print(f"Warning: {split_name}.txt not found. Skipping {split_name} processing.")
        return
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Track statistics
    total_images = 0
    identities = set()
    
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Extract session/path, identity, and attributes
            session_path = parts[0]
            identity = parts[1]
            identities.add(identity)
            
            # Determine camera ID from session
            camera_id = 1  # Default
            for session_name, cam_id in camera_mapping.items():
                if session_name in session_path:
                    camera_id = cam_id
                    break
            
            # Find source directory in session folders
            src_dir = None
            possible_paths = [
                os.path.join(download_path, session_path),
                os.path.join(download_path, session_path.replace('_', '/'))
            ]
            
            # Search in session folders
            for path in possible_paths:
                if os.path.isdir(path):
                    src_dir = path
                    break
            
            if not src_dir:
                # Fallback: search all session folders
                for session in ['session1', 'session2', 'session3']:
                    candidate = os.path.join(download_path, session, os.path.basename(session_path))
                    if os.path.isdir(candidate):
                        src_dir = candidate
                        camera_id = camera_mapping.get(session, 1)
                        break
            
            if not src_dir:
                print(f"Warning: Directory not found for {session_path}")
                continue
            
            # Get all images in the directory
            image_files = sorted(glob.glob(os.path.join(src_dir, '*.jpg')))
            if not image_files:
                print(f"Warning: No JPG images found in {src_dir}")
                continue
            
            # Copy all images with Market1501 naming convention
            # Images go directly into target_dir (no subdirectories)
            for idx, img_path in enumerate(image_files):
                # Create Market1501 style filename
                new_filename = create_market1501_filename(
                    identity, 
                    camera_id, 
                    sequence_num=idx+1,
                    frame_num=0
                )
                
                # Copy file with new name directly to target directory
                dst_path = os.path.join(target_dir, new_filename)
                copyfile(img_path, dst_path)
                total_images += 1
    
    print(f"Processed {split_name}: {len(identities)} identities, {total_images} total images")

# Create directory structure
bounding_box_train = os.path.join(save_path, 'bounding_box_train')
bounding_box_test = os.path.join(save_path, 'bounding_box_test')
query_dir = os.path.join(save_path, 'query')

# Process splits
print("Processing query set...")
process_split('query', query_dir)

print("\nProcessing gallery set...")
process_split('gallery', bounding_box_test)

print("\nProcessing training set...")
process_split('train', bounding_box_train)

# Create a simple train/val split if needed
# TAO can use bounding_box_train directly, but if you want separate val:
print("\nCreating validation split...")
val_dir = os.path.join(save_path, 'bounding_box_val')
os.makedirs(val_dir, exist_ok=True)

# Move 20% of training images to validation
train_images = sorted([f for f in os.listdir(bounding_box_train) if f.endswith('.jpg')])
val_split_ratio = 0.2
val_count = int(len(train_images) * val_split_ratio)

# Group by identity and move some images from each identity
identity_images = {}
for img in train_images:
    identity = img.split('_')[0]
    if identity not in identity_images:
        identity_images[identity] = []
    identity_images[identity].append(img)

# Move images to validation
val_images_moved = 0
for identity, images in identity_images.items():
    # Move at least 1 image per identity to validation
    num_to_move = max(1, int(len(images) * val_split_ratio))
    for i in range(min(num_to_move, len(images))):
        src = os.path.join(bounding_box_train, images[i])
        dst = os.path.join(val_dir, images[i])
        os.rename(src, dst)
        val_images_moved += 1

print(f"Moved {val_images_moved} images to validation set")

print("\nCCVID dataset processing completed!")
print(f"Final structure in: {save_path}")
print("\nFor TAO training, use these paths in your YAML:")
print(f"  train_dataset: {bounding_box_train}")
print(f"  val_dataset: {val_dir}")
print(f"  query_dataset: {query_dir}")
print(f"  test_dataset: {bounding_box_test}")