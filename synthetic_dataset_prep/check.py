from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import re

# =========================
# Kontrol edilecek klasör
# =========================
input_dir = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\sampled_crops")

# Rapor klasörü
report_dir = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\preprocess_reports")
report_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Beklenen dosya formatı
# Örnek:
# 0018_c3s8_000001_00.jpg
# =========================
filename_pattern = re.compile(
    r"^(?P<id>\d{4})_c(?P<camera>\d+)s(?P<scene>\d+)_(?P<frame>\d{6})_(?P<variant>\d{2})(?:_dup\d+)?\.jpg$"
)

# =========================
# Crop size eşikleri
# Bunlar sadece raporlama için.
# Dosya silmez.
# =========================
MIN_WIDTH_WARN = 10
MIN_HEIGHT_WARN = 10
MAX_RATIO_WARN = 8.0  # Çok ince/uzun crop kontrolü

image_files = sorted(input_dir.glob("*.jpg"))

id_counter = Counter()
bad_name_files = []
size_records = []
suspicious_size_files = []
read_error_files = []

camera_counter = Counter()
scene_counter = Counter()
variant_counter = Counter()

for img_path in image_files:
    match = filename_pattern.match(img_path.name)

    if not match:
        bad_name_files.append(img_path.name)
        continue

    file_id = match.group("id")
    camera_id = match.group("camera")
    scene_id = match.group("scene")
    variant_id = match.group("variant")

    id_counter[file_id] += 1
    camera_counter[camera_id] += 1
    scene_counter[scene_id] += 1
    variant_counter[variant_id] += 1

    try:
        with Image.open(img_path) as img:
            w, h = img.size

        ratio = max(w / h, h / w) if w > 0 and h > 0 else 999

        size_records.append((img_path.name, file_id, w, h, ratio))

        if w < MIN_WIDTH_WARN or h < MIN_HEIGHT_WARN or ratio > MAX_RATIO_WARN:
            suspicious_size_files.append((img_path.name, file_id, w, h, ratio))

    except Exception as e:
        read_error_files.append((img_path.name, str(e)))


# =========================
# 1) ID distribution raporu
# =========================
id_report_path = report_dir / "id_distribution.txt"

with open(id_report_path, "w", encoding="utf-8") as f:
    f.write("ID DISTRIBUTION REPORT\n")
    f.write("======================\n\n")
    f.write(f"Total valid images: {sum(id_counter.values())}\n")
    f.write(f"Unique ID count   : {len(id_counter)}\n\n")

    f.write("ID image counts:\n")
    f.write("----------------\n")

    for file_id, count in id_counter.most_common():
        f.write(f"{file_id}: {count}\n")

    f.write("\nIDs with very few images <= 3:\n")
    f.write("------------------------------\n")

    few_ids = [(file_id, count) for file_id, count in id_counter.items() if count <= 3]

    if few_ids:
        for file_id, count in sorted(few_ids, key=lambda x: x[1]):
            f.write(f"{file_id}: {count}\n")
    else:
        f.write("None\n")


# =========================
# 2) Bad filename raporu
# =========================
bad_name_report_path = report_dir / "bad_name_files.txt"

with open(bad_name_report_path, "w", encoding="utf-8") as f:
    f.write("BAD NAME FILES REPORT\n")
    f.write("=====================\n\n")
    f.write(f"Total jpg files checked : {len(image_files)}\n")
    f.write(f"Bad filename count      : {len(bad_name_files)}\n\n")

    if bad_name_files:
        for name in bad_name_files:
            f.write(name + "\n")
    else:
        f.write("No bad filename found.\n")


# =========================
# 3) Crop size raporu
# =========================
size_report_path = report_dir / "crop_size_report.txt"

with open(size_report_path, "w", encoding="utf-8") as f:
    f.write("CROP SIZE REPORT\n")
    f.write("================\n\n")
    f.write(f"Total valid size records : {len(size_records)}\n")
    f.write(f"Suspicious size count    : {len(suspicious_size_files)}\n")
    f.write(f"Read error count         : {len(read_error_files)}\n\n")

    if size_records:
        widths = [r[2] for r in size_records]
        heights = [r[3] for r in size_records]
        ratios = [r[4] for r in size_records]

        f.write("General statistics:\n")
        f.write("-------------------\n")
        f.write(f"Min width   : {min(widths)}\n")
        f.write(f"Max width   : {max(widths)}\n")
        f.write(f"Min height  : {min(heights)}\n")
        f.write(f"Max height  : {max(heights)}\n")
        f.write(f"Min ratio   : {min(ratios):.2f}\n")
        f.write(f"Max ratio   : {max(ratios):.2f}\n\n")

    f.write("Suspicious crop sizes:\n")
    f.write("----------------------\n")

    if suspicious_size_files:
        for name, file_id, w, h, ratio in suspicious_size_files:
            f.write(f"{name} | id={file_id} | size={w}x{h} | ratio={ratio:.2f}\n")
    else:
        f.write("No suspicious crop size found.\n")

    f.write("\nRead errors:\n")
    f.write("------------\n")

    if read_error_files:
        for name, err in read_error_files:
            f.write(f"{name} | error={err}\n")
    else:
        f.write("No read errors.\n")


# =========================
# Ek küçük özet: camera / scene / variant dağılımı
# =========================
extra_report_path = report_dir / "camera_scene_variant_distribution.txt"

with open(extra_report_path, "w", encoding="utf-8") as f:
    f.write("CAMERA / SCENE / VARIANT DISTRIBUTION\n")
    f.write("=====================================\n\n")

    f.write("Camera distribution:\n")
    f.write("--------------------\n")
    for cam, count in sorted(camera_counter.items(), key=lambda x: int(x[0])):
        f.write(f"c{cam}: {count}\n")

    f.write("\nScene distribution:\n")
    f.write("-------------------\n")
    for scene, count in sorted(scene_counter.items(), key=lambda x: int(x[0])):
        f.write(f"s{scene}: {count}\n")

    f.write("\nVariant distribution:\n")
    f.write("---------------------\n")
    for variant, count in sorted(variant_counter.items(), key=lambda x: int(x[0])):
        f.write(f"{variant}: {count}\n")


# =========================
# Konsol özeti
# =========================
print("\n========== CHECK ÖZET ==========")
print(f"Kontrol edilen jpg sayısı       : {len(image_files)}")
print(f"Formatı doğru dosya sayısı      : {sum(id_counter.values())}")
print(f"Formatı bozuk dosya sayısı      : {len(bad_name_files)}")
print(f"Unique ID sayısı                : {len(id_counter)}")
print(f"Şüpheli crop size sayısı        : {len(suspicious_size_files)}")
print(f"Okuma hatası sayısı             : {len(read_error_files)}")
print("--------------------------------")
print(f"ID raporu                       : {id_report_path}")
print(f"Bozuk isim raporu               : {bad_name_report_path}")
print(f"Crop size raporu                : {size_report_path}")
print(f"Camera/scene/variant raporu     : {extra_report_path}")
print("================================")