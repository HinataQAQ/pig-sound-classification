from pathlib import Path
import hashlib
import argparse
import pandas as pd


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def file_hash(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path, split_name: str):
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            lower = str(p).lower()
            if "scream" in lower:
                label = "scream"
            elif "cough" in lower:
                label = "cough"
            else:
                label = "unknown"

            rows.append({
                "split": split_name,
                "filepath": str(p.resolve()),
                "filename": p.name,
                "label": label,
                "md5": file_hash(p),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    train_rows = collect(Path(args.train_dir), "train")
    test_rows = collect(Path(args.test_dir), "test")

    df = pd.DataFrame(train_rows + test_rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("[COUNTS]")
    print(df.groupby(["split", "label"]).size())

    train_hash = set(df[df["split"] == "train"]["md5"])
    test_hash = set(df[df["split"] == "test"]["md5"])
    hash_overlap = train_hash & test_hash

    train_names = set(df[df["split"] == "train"]["filename"])
    test_names = set(df[df["split"] == "test"]["filename"])
    name_overlap = train_names & test_names

    print("\n[DUPLICATES]")
    print("filename overlap =", len(name_overlap))
    print("content md5 overlap =", len(hash_overlap))

    if len(hash_overlap) > 0:
        print("\n[EXAMPLE DUPLICATE HASHES]")
        dup = df[df["md5"].isin(hash_overlap)].sort_values("md5")
        print(dup.head(20)[["split", "label", "filename", "filepath", "md5"]])


if __name__ == "__main__":
    main()