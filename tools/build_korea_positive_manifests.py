from pathlib import Path
import csv
import random

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

SRC = {
    "dry_cough": ROOT / "data" / "dry_cough",
    "abdominal_cough": ROOT / "data" / "abdominal_cough",
}

OUT_DIR = ROOT / "data" / "manifests_korea"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 3407
TRAIN_PER_SUBTYPE = 2000
HOLDOUT_PER_SUBTYPE = 500

def collect_rows():
    rows = []
    for subtype, folder in SRC.items():
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        for p in files:
            rows.append({
                "filepath": str(p.resolve()),
                "label": "cough",
                "source": "korea",
                "subtype": subtype,
            })
    return rows

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    rng = random.Random(SEED)
    rows = collect_rows()

    by_subtype = {}
    for r in rows:
        by_subtype.setdefault(r["subtype"], []).append(r)

    train_rows = []
    holdout_rows = []
    unused_rows = []

    for subtype, items in by_subtype.items():
        rng.shuffle(items)

        n_hold = min(HOLDOUT_PER_SUBTYPE, len(items))
        n_train = min(TRAIN_PER_SUBTYPE, max(0, len(items) - n_hold))

        hold = items[:n_hold]
        train = items[n_hold:n_hold + n_train]
        unused = items[n_hold + n_train:]

        holdout_rows.extend(hold)
        train_rows.extend(train)
        unused_rows.extend(unused)

        print(f"{subtype}: total={len(items)} holdout={len(hold)} train={len(train)} unused={len(unused)}")

    fields = ["filepath", "label", "source", "subtype"]
    write_csv(OUT_DIR / "korea_cough_train_pos.csv", train_rows, fields)
    write_csv(OUT_DIR / "korea_cough_holdout_pos.csv", holdout_rows, fields)
    write_csv(OUT_DIR / "korea_cough_unused.csv", unused_rows, fields)

    # 生成和现有 train.csv 合并后的 manifest（只保留 filepath,label，兼容当前主线）
    base_train = ROOT / "data" / "manifests_cough_silence" / "train.csv"
    merged_out = OUT_DIR / "train_with_korea_pos.csv"

    merged_rows = []
    with open(base_train, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            merged_rows.append({
                "filepath": r["filepath"],
                "label": r["label"],
            })

    for r in train_rows:
        merged_rows.append({
            "filepath": r["filepath"],
            "label": "cough",
        })

    write_csv(merged_out, merged_rows, ["filepath", "label"])

    print(f"[OK] wrote -> {OUT_DIR / 'korea_cough_train_pos.csv'}")
    print(f"[OK] wrote -> {OUT_DIR / 'korea_cough_holdout_pos.csv'}")
    print(f"[OK] wrote -> {OUT_DIR / 'train_with_korea_pos.csv'}")

if __name__ == "__main__":
    main()