from pathlib import Path

# Ana dataset klasörünü buraya yaz
ROOT_DIR = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\sahneler\S2")

# Önce True bırak: sadece rapor verir, silmez.
# Kontrol ettikten sonra False yap.
DRY_RUN = False

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def collect_files(root_dir: Path):
    image_files = {}
    json_files = {}

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        # meta jsonları sayma / eşleştirme dışı bırak
        if path.name.endswith("_meta.json"):
            continue

        if suffix in IMAGE_EXTENSIONS:
            image_files[path.with_suffix("").as_posix()] = path

        elif suffix == ".json":
            json_files[path.with_suffix("").as_posix()] = path

    return image_files, json_files


def count_all_relevant_files(root_dir: Path):
    image_count = 0
    json_count = 0

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if path.name.endswith("_meta.json"):
            continue

        if suffix in IMAGE_EXTENSIONS:
            image_count += 1
        elif suffix == ".json":
            json_count += 1

    return image_count, json_count


def main():
    print(f"[INFO] Root dir: {ROOT_DIR}")

    before_image_count, before_json_count = count_all_relevant_files(ROOT_DIR)

    image_files, json_files = collect_files(ROOT_DIR)

    image_stems = set(image_files.keys())
    json_stems = set(json_files.keys())

    matched_stems = image_stems & json_stems

    unmatched_images = sorted(image_stems - json_stems)
    unmatched_jsons = sorted(json_stems - image_stems)

    print("\n========== BEFORE CLEANING ==========")
    print(f"Images before        : {before_image_count}")
    print(f"JSON files before    : {before_json_count}")
    print(f"Total files before   : {before_image_count + before_json_count}")
    print("=====================================")

    print("\n========== MATCH SUMMARY ==========")
    print(f"Matched pairs        : {len(matched_stems)}")
    print(f"Unmatched images     : {len(unmatched_images)}")
    print(f"Unmatched JSON files : {len(unmatched_jsons)}")
    print("===================================")

    if unmatched_images:
        print("\n[UNMATCHED IMAGES - will delete]")
        for stem in unmatched_images:
            print(image_files[stem])

    if unmatched_jsons:
        print("\n[UNMATCHED JSONS - will delete]")
        for stem in unmatched_jsons:
            print(json_files[stem])

    if DRY_RUN:
        print("\n[DRY RUN] Hiçbir dosya silinmedi.")
        print("Silmek için DRY_RUN = False yapıp tekrar çalıştır.")

        print("\n========== DRY RUN ESTIMATE ==========")
        print(f"Images after estimate      : {before_image_count - len(unmatched_images)}")
        print(f"JSON files after estimate  : {before_json_count - len(unmatched_jsons)}")
        print(f"Total files after estimate : {(before_image_count + before_json_count) - (len(unmatched_images) + len(unmatched_jsons))}")
        print("======================================")

        return

    deleted_images = 0
    deleted_jsons = 0

    for stem in unmatched_images:
        path = image_files[stem]
        try:
            path.unlink()
            deleted_images += 1
            print(f"[DELETED IMAGE] {path}")
        except Exception as e:
            print(f"[ERROR] Could not delete image {path}: {e}")

    for stem in unmatched_jsons:
        path = json_files[stem]
        try:
            path.unlink()
            deleted_jsons += 1
            print(f"[DELETED JSON] {path}")
        except Exception as e:
            print(f"[ERROR] Could not delete json {path}: {e}")

    after_image_count, after_json_count = count_all_relevant_files(ROOT_DIR)

    print("\n========== AFTER CLEANING ==========")
    print(f"Images after        : {after_image_count}")
    print(f"JSON files after    : {after_json_count}")
    print(f"Total files after   : {after_image_count + after_json_count}")
    print("====================================")

    print("\n========== DELETED ==========")
    print(f"Deleted images      : {deleted_images}")
    print(f"Deleted jsons       : {deleted_jsons}")
    print(f"Total deleted       : {deleted_images + deleted_jsons}")
    print("=============================")

    print("\n========== CHECK ==========")
    print(f"Before total        : {before_image_count + before_json_count}")
    print(f"After total         : {after_image_count + after_json_count}")
    print(f"Difference          : {(before_image_count + before_json_count) - (after_image_count + after_json_count)}")
    print("===========================")


if __name__ == "__main__":
    main()