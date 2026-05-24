import json
from pathlib import Path
from PIL import Image

# =========================
# Ana klasör yolları
# =========================
root_dir = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\sahneler\S8")

# Çıktı klasörü
output_dir = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\s8_cropped")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Çok küçük bbox filtreleri
# =========================
MIN_WIDTH = 5
MIN_HEIGHT = 5

# =========================
# ID seçimi
# character_id kullanmak için: "character"
# semantic_id kullanmak için : "semantic"
# =========================
ID_MODE = "character"

# =========================
# Sayaçlar
# =========================
total_scene_folders = 0
total_json = 0
total_saved = 0
total_skipped_small = 0
total_skipped_invalid = 0
total_missing_image = 0
total_error = 0
total_skipped_black = 0

# =========================
# sahneler altındaki S* klasörlerini gez
# =========================
scene_dirs = sorted([p for p in root_dir.glob("S*") if p.is_dir()])

print(f"[INFO] Bulunan sahne klasörü sayısı: {len(scene_dirs)}")

for input_dir in scene_dirs:
    total_scene_folders += 1
    json_files = sorted(input_dir.glob("*.json"))

    print(f"\n[SCENE] {input_dir.name} | JSON sayısı: {len(json_files)}")

    for json_path in json_files:
        total_json += 1

        try:
            # JSON oku
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # JSON içindeki image_file varsa onu kullan
            image_name = data.get("image_file", json_path.with_suffix(".jpg").name)
            image_path = input_dir / image_name

            # Eğer image_file bulunamazsa, JSON ile aynı isimli jpg dene
            if not image_path.exists():
                image_path = json_path.with_suffix(".jpg")

            # Görsel yoksa geç
            if not image_path.exists():
                print(f"[MISSING IMAGE] scene={input_dir.name} | json={json_path.name}")
                total_missing_image += 1
                continue

            # Görseli aç
            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            # Metadata
            camera_id = int(data["camera_id"])
            sequence_id = int(data["sequence_id"])
            frame_id = int(data["frame_id"])
            variant_id = int(data["variant_id"])

            boxes = data["annotations"]["boxes"]

            for box in boxes:
                raw_character_id = box.get("character_id", "NO_CHARACTER_ID")
                semantic_id = box.get("semantic_id", None)

                # =========================
                # Dosya adındaki insan ID seçimi
                # =========================
                if ID_MODE == "character":
                    id_num = int(str(raw_character_id).replace("_", ""))
                elif ID_MODE == "semantic":
                    id_num = int(semantic_id)
                else:
                    raise ValueError("ID_MODE sadece 'character' veya 'semantic' olabilir.")

                x_min = int(box["x_min"])
                y_min = int(box["y_min"])
                x_max = int(box["x_max"])
                y_max = int(box["y_max"])

                # Görsel sınırları içine al
                x_min = max(0, min(x_min, img_w))
                y_min = max(0, min(y_min, img_h))
                x_max = max(0, min(x_max, img_w))
                y_max = max(0, min(y_max, img_h))

                width = x_max - x_min
                height = y_max - y_min

                # Geçersiz bbox kontrolü
                if width <= 0 or height <= 0:
                    print(
                        f"[SKIP INVALID] scene={input_dir.name} | "
                        f"file={json_path.name} | "
                        f"character_id={raw_character_id} | "
                        f"semantic_id={semantic_id} | "
                        f"bbox=({x_min}, {y_min}, {x_max}, {y_max}) | "
                        f"size={width}x{height}"
                    )
                    total_skipped_invalid += 1
                    continue

                # Aşırı ince / anlamsız bbox kontrolü
                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    print(
                        f"[SKIP TOO SMALL] scene={input_dir.name} | "
                        f"file={json_path.name} | "
                        f"character_id={raw_character_id} | "
                        f"semantic_id={semantic_id} | "
                        f"bbox=({x_min}, {y_min}, {x_max}, {y_max}) | "
                        f"size={width}x{height}"
                    )
                    total_skipped_small += 1
                    continue

                crop = img.crop((x_min, y_min, x_max, y_max))

                # =========================
                # Tamamen siyah crop kontrolü
                # =========================
                extrema = crop.getextrema()
                # RGB için örnek siyah: ((0, 0), (0, 0), (0, 0))

                is_fully_black = all(channel_max == 0 for channel_min, channel_max in extrema)

                if is_fully_black:
                    total_skipped_black += 1
                    print(
                        f"[SKIP FULL BLACK] scene={input_dir.name} | "
                        f"file={json_path.name} | "
                        f"bbox=({x_min}, {y_min}, {x_max}, {y_max}) | "
                        f"size={width}x{height} | "
                        f"extrema={extrema}"
                    )
                    continue

                # İsim formatı:
                out_name = (
                    f"{id_num:04d}"
                    f"_c{camera_id}s{sequence_id}"
                    f"_{frame_id:06d}"
                    f"_{variant_id:02d}.jpg"
                )

                out_path = output_dir / out_name

                # Aynı isim varsa üstüne yazmamak için suffix ekle
                if out_path.exists():
                    stem = out_path.stem
                    suffix = out_path.suffix
                    i = 1

                    while True:
                        new_out_path = output_dir / f"{stem}_dup{i}{suffix}"
                        if not new_out_path.exists():
                            out_path = new_out_path
                            break
                        i += 1

                crop.save(out_path, quality=95)
                total_saved += 1

                print(
                    f"[OK] {out_path.name} kaydedildi | "
                    f"scene={input_dir.name} | "
                    f"source={image_path.name} | "
                    f"character_id={raw_character_id} | "
                    f"semantic_id={semantic_id} | "
                    f"bbox=({x_min}, {y_min}, {x_max}, {y_max}) | "
                    f"size={width}x{height}"
                )

        except Exception as e:
            print(f"[ERROR] scene={input_dir.name} | json={json_path.name} | hata={e}")
            total_error += 1


# =========================
# Özet
# =========================
print("\n========== ÖZET ==========")
print(f"İşlenen sahne klasörü     : {total_scene_folders}")
print(f"İşlenen JSON sayısı       : {total_json}")
print(f"Kaydedilen crop sayısı    : {total_saved}")
print(f"Çok küçük diye atlanan    : {total_skipped_small}")
print(f"Geçersiz bbox atlanan     : {total_skipped_invalid}")
print(f"Görsel bulunamayan JSON   : {total_missing_image}")
print(f"Hata alınan JSON          : {total_error}")
print(f"Tam siyah diye atlanan    : {total_skipped_black}")
print("==========================")