from pathlib import Path
import re
import shutil
import argparse
import pandas as pd


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

LABEL_MAP = {
    "0": "calm_grunt",
    "1": "feeding",
    "2": "frightened_stress",
    "3": "anxious_stress",
}


def parse_label_from_name(path: Path):
    """
    支持类似：
    639-2.wav
    001-0.wav
    xxx-3.wav
    """
    stem = path.stem
    m = re.search(r"-(\d+)$", stem)
    if not m:
        return None
    lab = m.group(1)
    return LABEL_MAP.get(lab)


def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        dst = dst_dir / f"{src.stem}__dup{src.suffix}"

    shutil.copy2(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--copy_files", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    rows = []
    unknown = []

    files = [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]

    print("[INFO] total audio =", len(files))

    for p in files:
        cls = parse_label_from_name(p)

        if cls is None:
            unknown.append(str(p))
            continue

        if args.copy_files:
            review_path = safe_copy(p, out_dir / cls)
        else:
            review_path = ""

        rows.append({
            "filepath": str(p.resolve()),
            "label_raw": p.stem.split("-")[-1],
            "class_name": cls,
            "review_path": str(review_path) if review_path else "",
        })

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sow_labeled_index.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"filepath": unknown}).to_csv(
        out_dir / "sow_unknown_label.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[OK] wrote ->", out_dir / "sow_labeled_index.csv")
    print("[OK] wrote ->", out_dir / "sow_unknown_label.csv")

    print("\n[COUNTS]")
    print(df["class_name"].value_counts())


if __name__ == "__main__":
    main()