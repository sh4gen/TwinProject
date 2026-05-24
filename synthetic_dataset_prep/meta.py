from pathlib import Path

# Ana folder pathini buraya yaz
ROOT_DIR = Path(r"C:\Users\askin\OneDrive\Masaüstü\dataset_clean\sahneler\S8")

# Önce True bırak, ne sileceğini gör.
# Emin olunca False yapıp tekrar çalıştır.
DRY_RUN = False

def main():
    meta_files = sorted(ROOT_DIR.rglob("*_meta.json"))

    print(f"[INFO] Root folder: {ROOT_DIR}")
    print(f"[INFO] Found meta JSON files: {len(meta_files)}")

    if not meta_files:
        print("[INFO] No *_meta.json files found.")
        return

    print("\nFiles to delete:")
    for file_path in meta_files:
        print(file_path)

    if DRY_RUN:
        print("\n[DRY RUN] Hiçbir dosya silinmedi.")
        print("Silmek için DRY_RUN = False yapıp tekrar çalıştır.")
        return

    deleted_count = 0

    for file_path in meta_files:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"[DELETED] {file_path}")
        except Exception as e:
            print(f"[ERROR] Could not delete {file_path}: {e}")

    print(f"\n[DONE] Deleted meta JSON files: {deleted_count}")

if __name__ == "__main__":
    main()