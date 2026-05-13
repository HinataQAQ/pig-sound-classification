from pathlib import Path
import random
import csv

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
OUT = ROOT / "data" / "manifests_korea_subtype"
OUT.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SEED = 3407

# ===== 这里改成你现在真实的目录 =====
SRC = {
    "dry_cough": ROOT / "data" / "external" / "korea_raw" / "dry_cough",
    "abdominal_cough": ROOT / "data" / "external" / "korea_raw" / "abdominal_cough",
}

def collect(folder: Path, label: str):
    if not folder.exists():
        print(f"[WARN] folder not found: {folder}")
        return []

    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    print(f"[INFO] {label}: found {len(files)} files in {folder}")

    rows = []
    for p in files:
        rows.append({
            "filepath": str(p.resolve()),
            "label": label,
        })
    return rows

def split_class(rows, train_ratio=0.8, val_ratio=0.1):
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    return train, val, test

def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(rows)

def main():
    rng = random.Random(SEED)

    dry_rows = collect(SRC["dry_cough"], "dry_cough")
    abd_rows = collect(SRC["abdominal_cough"], "abdominal_cough")

    if len(dry_rows) == 0:
        raise RuntimeError("dry_cough folder has 0 usable audio files. Check path and file extensions.")
    if len(abd_rows) == 0:
        raise RuntimeError("abdominal_cough folder has 0 usable audio files. Check path and file extensions.")

    rng.shuffle(dry_rows)
    rng.shuffle(abd_rows)

    dry_tr, dry_va, dry_te = split_class(dry_rows)
    abd_tr, abd_va, abd_te = split_class(abd_rows)

    train = dry_tr + abd_tr
    val = dry_va + abd_va
    test = dry_te + abd_te

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    write_csv(OUT / "train.csv", train)
    write_csv(OUT / "val.csv", val)
    write_csv(OUT / "test.csv", test)

    print("[OK] wrote ->", OUT / "train.csv")
    print("[OK] wrote ->", OUT / "val.csv")
    print("[OK] wrote ->", OUT / "test.csv")
    print("train =", len(train))
    print("val   =", len(val))
    print("test  =", len(test))

if __name__ == "__main__":
    main()