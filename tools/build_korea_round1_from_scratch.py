from pathlib import Path
import csv
import random

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

SRC = {
    "dry_cough": ROOT / "data" / "external" / "korea_raw" / "dry_cough",
    "abdominal_cough": ROOT / "data" / "external" / "korea_raw" / "abdominal_cough",
}

OUT_DIR = ROOT / "data" / "manifests_korea"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 3407

# 第一轮先不要全灌进去
TRAIN_PER_SUBTYPE = 500
HOLDOUT_PER_SUBTYPE = 500

def collect_files(folder: Path):
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=lambda x: str(x).lower()
    )

def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    rng = random.Random(SEED)

    rich_all = []
    train_pos = []
    holdout_pos = []
    unused = []

    for subtype, folder in SRC.items():
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = collect_files(folder)
        if len(files) == 0:
            raise RuntimeError(f"No audio files found in {folder}")

        rng.shuffle(files)

        n_hold = min(HOLDOUT_PER_SUBTYPE, len(files))
        n_train = min(TRAIN_PER_SUBTYPE, max(0, len(files) - n_hold))

        hold_files = files[:n_hold]
        train_files = files[n_hold:n_hold + n_train]
        unused_files = files[n_hold + n_train:]

        for p in files:
            rich_all.append({
                "filepath": str(p.resolve()),
                "label": "cough",
                "source": "korea",
                "subtype": subtype,
            })

        for p in train_files:
            train_pos.append({
                "filepath": str(p.resolve()),
                "label": "cough",
            })

        for p in hold_files:
            holdout_pos.append({
                "filepath": str(p.resolve()),
                "label": "cough",
            })

        for p in unused_files:
            unused.append({
                "filepath": str(p.resolve()),
                "label": "cough",
                "source": "korea",
                "subtype": subtype,
            })

        print(f"{subtype}: total={len(files)} train={len(train_files)} holdout={len(hold_files)} unused={len(unused_files)}")

    write_csv(OUT_DIR / "korea_all_rich.csv", rich_all, ["filepath", "label", "source", "subtype"])
    write_csv(OUT_DIR / "korea_train_pos_round1.csv", train_pos, ["filepath", "label"])
    write_csv(OUT_DIR / "korea_holdout_pos_round1.csv", holdout_pos, ["filepath", "label"])
    write_csv(OUT_DIR / "korea_unused.csv", unused, ["filepath", "label", "source", "subtype"])

    # 合并到现有主训练集
    base_train_path = ROOT / "data" / "manifests_cough_silence" / "train.csv"
    if not base_train_path.exists():
        raise FileNotFoundError(f"Missing base train manifest: {base_train_path}")

    merged = []
    with open(base_train_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            merged.append({
                "filepath": row["filepath"],
                "label": row["label"],
            })

    merged.extend(train_pos)

    write_csv(OUT_DIR / "train_plus_korea_round1.csv", merged, ["filepath", "label"])

    print("[OK] wrote ->", OUT_DIR / "korea_all_rich.csv")
    print("[OK] wrote ->", OUT_DIR / "korea_train_pos_round1.csv")
    print("[OK] wrote ->", OUT_DIR / "korea_holdout_pos_round1.csv")
    print("[OK] wrote ->", OUT_DIR / "train_plus_korea_round1.csv")

if __name__ == "__main__":
    main()