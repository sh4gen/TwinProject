import os
import re
from PIL import Image

# --- CONFIGURE THIS ---
# Point this to the directory containing your train/test/query folders
root_dir = "/home/ika/yzlm/TwinProject/ReID_Experiments/LTCC_ReID/data"
# ----------------------

print(f"Starting to process files in: {root_dir}")

# Regular expression to capture the parts of the LTCC filename
# It captures (PersonID), (ClothID), (CameraID), and (FrameInfo)
ltcc_pattern = re.compile(r'^(\d+)_(\d+)_(c\d+)_(\d+)\.png$')

files_renamed = 0
files_failed = 0

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        match = ltcc_pattern.match(filename)
        if match:
            # Extract parts from the filename
            person_id = match.group(1)
            cloth_id = match.group(2) # We will ignore this for the new name
            camera_id = match.group(3)
            frame_info = match.group(4)

            # Create the new, TAO-compatible filename
            new_filename_stem = f"{person_id}_{camera_id}_f{frame_info}"
            new_filename_jpg = f"{new_filename_stem}.jpg"
            
            old_filepath = os.path.join(dirpath, filename)
            new_filepath = os.path.join(dirpath, new_filename_jpg)

            try:
                # Open the PNG image, convert to RGB, and save as JPG
                with Image.open(old_filepath) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(new_filepath, 'jpeg')
                
                # Remove the old PNG file after successful conversion
                os.remove(old_filepath)
                files_renamed += 1

            except Exception as e:
                print(f"!!! FAILED to process {old_filepath}: {e}")
                files_failed += 1

print(f"\nProcessing complete.")
print(f"Successfully renamed and converted {files_renamed} files.")
if files_failed > 0:
    print(f"Failed to process {files_failed} files. Please check the errors above.")
