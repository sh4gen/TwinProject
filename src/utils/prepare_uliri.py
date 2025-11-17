import os
import shutil
import re
import random
from collections import defaultdict
from PIL import Image

def create_market1501_structure(target_root):
    os.makedirs(os.path.join(target_root, 'bounding_box_train'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'bounding_box_test'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'query'), exist_ok=True)
    print("Created target directory structure.")

def convert_dataset(source_root, target_root, train_split_ratio=0.8, query_per_id=5):
    """
    Converts the ULIRI dataset to Market-1501 format with multiple query images per test ID.
    """
    target_train_dir = os.path.join(target_root, 'bounding_box_train')
    target_test_dir = os.path.join(target_root, 'bounding_box_test')
    target_query_dir = os.path.join(target_root, 'query')

    # 1. Identify and Split Person IDs
    all_person_dirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d)) and d.startswith('idt')]
    random.seed(42)
    random.shuffle(all_person_dirs)

    split_index = int(len(all_person_dirs) * train_split_ratio)
    train_person_dirs = all_person_dirs[:split_index]
    test_person_dirs = all_person_dirs[split_index:]

    print(f"Found {len(all_person_dirs)} total persons.")
    print(f"Splitting into {len(train_person_dirs)} training IDs and {len(test_person_dirs)} testing IDs.")

    # 2. Process training IDs
    frame_counters = defaultdict(int)
    for person_dir_name in train_person_dirs:
        person_id_match = re.search(r'\d+', person_dir_name)
        if not person_id_match:
            continue
        person_id = person_id_match.group(0)
        person_id_padded = person_id.zfill(4)
        source_person_path = os.path.join(source_root, person_dir_name)

        for filename in sorted(os.listdir(source_person_path)):
            if not filename.endswith('.png'):
                continue
            cam_match = re.search(r'_cam(\d+)_', filename)
            if not cam_match:
                continue
            camera_id_str = cam_match.group(1)
            camera_id_num = int(camera_id_str) + 1  # 1-based

            frame_counters[(person_id_padded, camera_id_num)] += 1
            frame_num = frame_counters[(person_id_padded, camera_id_num)]
            new_filename = f"{person_id_padded}_c{camera_id_num}s1_{frame_num:02d}_00.jpg"

            source_filepath = os.path.join(source_person_path, filename)
            target_filepath = os.path.join(target_train_dir, new_filename)

            try:
                with Image.open(source_filepath) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(target_filepath, 'JPEG')
            except Exception as e:
                print(f"Error converting {source_filepath}: {e}")
                continue

    # 3. Process test IDs (collect for query/gallery split)
    frame_counters.clear()
    test_id_to_images = defaultdict(list)  # person_id_padded -> list of (src_path, cam_id, new_filename)
    for person_dir_name in test_person_dirs:
        person_id_match = re.search(r'\d+', person_dir_name)
        if not person_id_match:
            continue
        person_id = person_id_match.group(0)
        person_id_padded = person_id.zfill(4)
        source_person_path = os.path.join(source_root, person_dir_name)

        for filename in sorted(os.listdir(source_person_path)):
            if not filename.endswith('.png'):
                continue
            cam_match = re.search(r'_cam(\d+)_', filename)
            if not cam_match:
                continue
            camera_id_str = cam_match.group(1)
            camera_id_num = int(camera_id_str) + 1  # 1-based

            frame_counters[(person_id_padded, camera_id_num)] += 1
            frame_num = frame_counters[(person_id_padded, camera_id_num)]
            new_filename = f"{person_id_padded}_c{camera_id_num}s1_{frame_num:02d}_00.jpg"

            source_filepath = os.path.join(source_person_path, filename)
            test_id_to_images[person_id_padded].append((source_filepath, camera_id_num, new_filename))

    # 4. For each test ID, select multiple query images, rest go to gallery
    total_query = 0
    total_gallery = 0
    for person_id, images in test_id_to_images.items():
        # Group images by camera
        cam_to_images = defaultdict(list)
        for src_path, cam_id, new_filename in images:
            cam_to_images[cam_id].append((src_path, new_filename))

        # Flatten all images for this ID
        all_images = []
        for img_list in cam_to_images.values():
            all_images.extend(img_list)

        # Shuffle for randomness
        random.shuffle(all_images)

        # Select up to query_per_id images as query (or all if less)
        num_query = min(query_per_id, len(all_images))
        query_images = all_images[:num_query]
        gallery_images = all_images[num_query:]

        # Save query images
        for src_path, new_filename in query_images:
            with Image.open(src_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(os.path.join(target_query_dir, new_filename), 'JPEG')
            total_query += 1

        # Save gallery images
        for src_path, new_filename in gallery_images:
            with Image.open(src_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(os.path.join(target_test_dir, new_filename), 'JPEG')
            total_gallery += 1

    print(f"Finished creating {total_query} query images and {total_gallery} gallery images.")

if __name__ == "__main__":
    # Set your parameters here
    SOURCE_DIRECTORY = "/home/ika/Downloads/Datasets/ULIRI"
    TARGET_DIRECTORY = "/home/ika/Downloads/Datasets/uliri_train_data_corrected"
    TRAIN_SPLIT_RATIO = 0.7  # 80% train, 20% test
    QUERY_PER_ID = 100         # Number of query images per test ID

    print(f"Source root directory: {os.path.abspath(SOURCE_DIRECTORY)}")
    print(f"Target root directory: {os.path.abspath(TARGET_DIRECTORY)}")

    create_market1501_structure(TARGET_DIRECTORY)
    convert_dataset(SOURCE_DIRECTORY, TARGET_DIRECTORY, train_split_ratio=TRAIN_SPLIT_RATIO, query_per_id=QUERY_PER_ID)

    print("\nConversion complete!")
    print(f"Your Market-1501 formatted dataset is ready at: {os.path.abspath(TARGET_DIRECTORY)}")