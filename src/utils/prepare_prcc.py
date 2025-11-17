import os
import shutil
import re
from collections import defaultdict

def create_market1501_structure(target_root):
    """Creates the necessary Market-1501 directory structure."""
    os.makedirs(os.path.join(target_root, 'bounding_box_train'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'bounding_box_test'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'query'), exist_ok=True)
    print("Created target directory structure.")

def process_rgb_dataset(source_root, target_root):
    """
    Main function to process the rgb directories and convert them to the Market-1501 format.
    """
    # --- Configuration ---
    # Maps camera folder names ('A', 'B', 'C') to numeric IDs
    camera_map = {'A': 1, 'B': 2, 'C': 3}
    
    # Keeps track of frame counts for each (person_id, camera_id) pair to avoid collisions
    frame_counters = defaultdict(int)

    # --- 1. Process Train and Validation sets (merging into bounding_box_train) ---
    print("\n--- Processing Train & Validation Sets (Source: rgb/train, rgb/val) ---")
    target_train_dir = os.path.join(target_root, 'bounding_box_train')
    
    for split in ['train', 'val']:
        source_dir = os.path.join(source_root, 'rgb', split)
        if not os.path.exists(source_dir):
            print(f"Warning: Directory not found, skipping: {source_dir}")
            continue
            
        print(f"Processing directory: {source_dir}")
        person_id_folders = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

        for person_id in person_id_folders:
            person_folder_path = os.path.join(source_dir, person_id)
            for filename in sorted(os.listdir(person_folder_path)):
                # Parse filenames like 'A_cropped_rgb220.jpg'
                match = re.match(r'([A-Z])_cropped_rgb(\d+)\.jpg', filename)
                if not match:
                    continue

                camera_char = match.group(1)
                camera_id = camera_map.get(camera_char)
                person_id_padded = person_id.zfill(4)
                
                # Generate new filename
                sequence_id = 1  # Assign a default sequence ID
                frame_counters[(person_id_padded, camera_id)] += 1
                frame_num = frame_counters[(person_id_padded, camera_id)]
                
                new_filename = f"{person_id_padded}_c{camera_id}s{sequence_id}_{frame_num:02d}_00.jpg"
                
                source_path = os.path.join(person_folder_path, filename)
                target_path = os.path.join(target_train_dir, new_filename)
                
                shutil.copy(source_path, target_path)
    print("Finished processing train/val sets.")

    # --- 2. Process Test set (splitting into Gallery and Query) ---
    print("\n--- Processing Test Set (Source: rgb/test) ---")
    source_test_dir = os.path.join(source_root, 'rgb', 'test')
    target_gallery_dir = os.path.join(target_root, 'bounding_box_test')
    target_query_dir = os.path.join(target_root, 'query')

    # This dictionary will store the first image we find for each person to use as the query.
    query_images_to_pick = {}

    if os.path.exists(source_test_dir):
        camera_folders = sorted([d for d in os.listdir(source_test_dir) if os.path.isdir(os.path.join(source_test_dir, d))])
        for camera_char in camera_folders:
            camera_id = camera_map.get(camera_char)
            camera_folder_path = os.path.join(source_test_dir, camera_char)
            person_id_folders = sorted([d for d in os.listdir(camera_folder_path) if os.path.isdir(os.path.join(camera_folder_path, d))])

            for person_id in person_id_folders:
                person_folder_path = os.path.join(camera_folder_path, person_id)
                for filename in sorted(os.listdir(person_folder_path)):
                    person_id_padded = person_id.zfill(4)
                    
                    # Generate new filename
                    sequence_id = 1
                    frame_counters[(person_id_padded, camera_id)] += 1
                    frame_num = frame_counters[(person_id_padded, camera_id)]
                    new_filename = f"{person_id_padded}_c{camera_id}s{sequence_id}_{frame_num:02d}_00.jpg"
                    
                    source_path = os.path.join(person_folder_path, filename)
                    
                    # A) Copy ALL test images to the gallery (bounding_box_test)
                    target_gallery_path = os.path.join(target_gallery_dir, new_filename)
                    shutil.copy(source_path, target_gallery_path)
                    
                    # B) Pick the FIRST image of each person to be the query image
                    if person_id_padded not in query_images_to_pick:
                        query_images_to_pick[person_id_padded] = {
                            "source_path": source_path,
                            "new_filename": new_filename
                        }
        
        # C) Now, copy the selected query images to the query folder
        print(f"Identified {len(query_images_to_pick)} unique persons in test set to create queries.")
        for person_id, info in query_images_to_pick.items():
            target_query_path = os.path.join(target_query_dir, info["new_filename"])
            shutil.copy(info["source_path"], target_query_path)

    else:
        print(f"Warning: Directory not found, skipping: {source_test_dir}")
    print("Finished processing test/gallery/query sets.")

if __name__ == "__main__":
    # Define your source and target directories
    # Source directory is the one containing the 'rgb' folder.
    SOURCE_DIRECTORY = "/home/ika/Downloads/Datasets/prcc" 
    
    # Target directory is where the 'data/market1501' structure will be created.
    TARGET_DIRECTORY = "prcc/train_data"
    
    print(f"Source root directory: {os.path.abspath(SOURCE_DIRECTORY)}")
    print(f"Target root directory: {os.path.abspath(TARGET_DIRECTORY)}")

    # 1. Create the target folder structure
    create_market1501_structure(TARGET_DIRECTORY)
    
    # 2. Process all files
    process_rgb_dataset(SOURCE_DIRECTORY, TARGET_DIRECTORY)
    
    print("\nConversion complete!")
    print(f"Your Market-1501 formatted dataset is ready at: {os.path.abspath(TARGET_DIRECTORY)}")
