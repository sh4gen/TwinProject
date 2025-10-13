import os
import shutil

def parse_ccvid_txt(txt_path):
    """
    Parse a CCVID split txt file.
    Returns a list of tuples: (video_path, person_id, clothes_label)
    """
    items = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                items.append(tuple(parts))
    return items

def get_camera_and_sequence(video_path):
    """
    Extract camera and sequence from video_path, e.g. session3/001_01
    Returns (camera_id, sequence_id)
    """
    # Example: session3/001_01
    session, vid = video_path.split('/')
    # Use session as sequence, vid as camera
    # session3 -> s3, 001_01 -> c1 (or c01)
    sequence_id = int(session.replace('session', ''))
    camera_id = int(vid.split('_')[1])  # 001_01 -> 01
    return camera_id, sequence_id

def convert_ccvid_to_market1501(ccvid_root, split_txt, out_dir, image_ext='.jpg'):
    """
    Convert a CCVID split to Market1501 format.
    """
    os.makedirs(out_dir, exist_ok=True)
    items = parse_ccvid_txt(split_txt)
    for item in items:
        video_path, person_id, clothes_label = item
        person_id = int(person_id)
        camera_id, sequence_id = get_camera_and_sequence(video_path)
        # Find all images for this video_path
        # Assume images are in ccvid_root/video_path/*.jpg
        img_dir = os.path.join(ccvid_root, video_path)
        if not os.path.isdir(img_dir):
            print(f"Warning: {img_dir} not found, skipping.")
            continue
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(image_ext)])
        for idx, img_file in enumerate(img_files):
            # Market1501 format: 0001_c1s1_01_00.jpg
            new_name = f"{person_id:04d}_c{camera_id}s{sequence_id}_{idx+1:02d}_00{image_ext}"
            src = os.path.join(img_dir, img_file)
            dst = os.path.join(out_dir, new_name)
            shutil.copy2(src, dst)
    print(f"Done: {out_dir}")

if __name__ == "__main__":
    ccvid_root = "/home/ika/Downloads/CCVID"  # Root directory of CCVID, containing session1, session2, ...
    out_root = "/home/ika/yzlm/TwinProject/CCVID/data"
    splits = [
        ("train.txt", "bounding_box_train"),
        ("gallery.txt", "bounding_box_test"),
        ("query.txt", "query"),
    ]
    for split_txt, out_dir in splits:
        split_txt_path = os.path.join(ccvid_root, split_txt)
        out_dir_path = os.path.join(out_root, out_dir)
        convert_ccvid_to_market1501(ccvid_root, split_txt_path, out_dir_path, image_ext='.jpg')