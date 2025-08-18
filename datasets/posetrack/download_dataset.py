import argparse
import os 
import requests
import shutil
import tarfile
import zipfile
from tqdm.auto import tqdm
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def download_posetrack_videos(download_path, video_source_url):
    archive_path = f"{video_source_url}/posetrack18_images.tar.a"

    os.makedirs(download_path, exist_ok=True)

    files = [97 + i for i in range(18)]
    for i, f in enumerate(files):
        file_letter = chr(f)
        file_name = f"posetrack18_images.tar.a{file_letter}" 
        save_path = os.path.join(download_path, file_name) 

        if os.path.exists(save_path):
            print(f"[{i+1}/{len(files)}] Atlanıyor: {file_name} zaten mevcut.")
            continue

        try:
            remote_url = f"{archive_path}{file_letter}" 
            print(f"[{i+1}/{len(files)}] İndiriliyor: {remote_url}")
            with requests.get(remote_url, stream=True, verify=False, timeout=30) as r:
                r.raise_for_status()
                total_length = int(r.headers.get("Content-Length", 0))

                with open(save_path, "wb") as f:
                    with tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"{file_name}") as raw:
                        shutil.copyfileobj(raw, f)

        except Exception as e:
            print(f"[HATA] {file_name} indirilemedi:\n{e}")

    print("Done")
    print("Merging splits")
    total_file = os.path.join(download_path, 'total.tar')
    if not os.path.exists(total_file):
        with open(total_file, 'wb') as fp:
            for f in tqdm(files):
                file_letter = chr(f) 
                file_name = f"posetrack18_images.tar.a{file_letter}" 
                save_path = os.path.join(download_path, file_name) 

                with open(save_path, 'rb') as read:
                    fp.write(read.read())

    print("Done")
    # Silme işlemi kaldırıldı
    return total_file


def download_annotations(download_save_path, download_path):
    anno_path = os.path.join(download_save_path, 'annotations.zip')

    if not os.path.exists(anno_path):
        print("Downloading annotations")
        with requests.get(download_path, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))

            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(anno_path, "wb") as out:
                    shutil.copyfileobj(raw, out)
    else:
        print("Annotations already downloaded")

    return anno_path 

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--save_path', type=str, default='data/PoseTrack21')
    parser.add_argument('--download_url', type=str, default='https://github.com/anDoer/PoseTrack21/releases/download/v0.1/posetrack21_annotations.zip')
    parser.add_argument('--video_source_url', type=str, default='https://posetrack.net/posetrack18-data/')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    download_path = 'downloads'
    archive_path = download_posetrack_videos(download_path, video_source_url=args.video_source_url)
    annotation_path = download_annotations(download_path, args.download_url)
    
    print("Unpacking Dataset")

    with tarfile.open(archive_path) as archive_fp:
        def is_within_directory(directory, target):
        	
        	abs_directory = os.path.abspath(directory)
        	abs_target = os.path.abspath(target)
        
        	prefix = os.path.commonprefix([abs_directory, abs_target])
        	
        	return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
        	for member in tar.getmembers():
        		member_path = os.path.join(path, member.name)
        		if not is_within_directory(path, member_path):
        			raise Exception("Attempted Path Traversal in Tar File")
        
        	tar.extractall(path, members, numeric_owner=numeric_owner) 
        	
        
        safe_extract(archive_fp, save_path)
    # ŞİFRE BURADA DOĞRUDAN TANIMLANIYOR
    # Manuel olarak şifrenizi buraya girin.
    annotation_password = '------------------' 
    
    # Zip dosyasını çıkarın
    with zipfile.ZipFile(annotation_path, 'r') as zip_fp:
        zip_fp.extractall(save_path, pwd=annotation_password.encode('utf-8'))
    
    print("Dataset successfully unpacked.")
    
