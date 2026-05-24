import random
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# Klasör yolları
# =========================
input_dir = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\s8_cropped")

output_dir = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\sampled_crops")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Ayarlar
# =========================
STEP = 5
SEED = 42

random.seed(SEED)

# =========================
# Dosyaları gruplama
# Format:
# 0018_c3s8_000001_00.jpg
# =========================
groups = defaultdict(list)

image_files = sorted(input_dir.glob("*.jpg"))

for img_path in image_files:
    stem = img_path.stem
    parts = stem.split("_")

    if len(parts) != 4:
        print(f"[SKIP NAME FORMAT] {img_path.name}")
        continue

    person_or_semantic_id = parts[0]   # 0018
    cam_scene = parts[1]               # c3s8
    frame_id = parts[2]                # 000001
    variant_id = parts[3]              # 00

    try:
        variant_num = int(variant_id)
    except ValueError:
        print(f"[SKIP VARIANT ERROR] {img_path.name}")
        continue

    group_key = f"{person_or_semantic_id}_{cam_scene}_{frame_id}"
    groups[group_key].append((variant_num, img_path))


# =========================
# Her grupta 5'lik aralıktan 1 random seç
# =========================
total_groups = 0
total_selected = 0
total_copied = 0

for group_key, items in groups.items():
    total_groups += 1

    items = sorted(items, key=lambda x: x[0])

    selected_files = []

    for i in range(0, len(items), STEP):
        chunk = items[i:i + STEP]

        if not chunk:
            continue

        selected_variant, selected_path = random.choice(chunk)
        selected_files.append(selected_path)

    for selected_path in selected_files:
        out_path = output_dir / selected_path.name

        if out_path.exists():
            stem = out_path.stem
            suffix = out_path.suffix
            dup_idx = 1

            while True:
                new_out_path = output_dir / f"{stem}_dup{dup_idx}{suffix}"
                if not new_out_path.exists():
                    out_path = new_out_path
                    break
                dup_idx += 1

        shutil.copy2(selected_path, out_path)
        total_copied += 1

    total_selected += len(selected_files)

    print(
        f"[OK] group={group_key} | "
        f"total={len(items)} | "
        f"selected={len(selected_files)}"
    )


print("\n========== ÖZET ==========")
print(f"Toplam crop sayısı       : {len(image_files)}")
print(f"Toplam grup sayısı       : {total_groups}")
print(f"Seçilen crop sayısı      : {total_selected}")
print(f"Kopyalanan crop sayısı   : {total_copied}")
print(f"Output klasörü           : {output_dir}")
print("==========================")