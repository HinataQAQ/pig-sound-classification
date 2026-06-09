from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}


def infer_path_col(df: pd.DataFrame) -> str:
    for c in ["path", "filepath", "file", "wav", "audio_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column from columns: {list(df.columns)}")


def resolve_path(p: str) -> Path:
    q = Path(str(p))
    if q.is_absolute():
        return q
    return ROOT / q


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_split(name: str, csv_path: str):
    df = pd.read_csv(csv_path)
    path_col = infer_path_col(df)

    rows = []
    for _, r in df.iterrows():
        p = resolve_path(r[path_col])
        if not p.exists():
            raise FileNotFoundError(p)
        rows.append({
            "split": name,
            "path": p,
            "rel": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    rows = []
    rows.extend(load_split("train", args.train))
    rows.extend(load_split("val", args.val))
    rows.extend(load_split("test", args.test))

    print(f"[INFO] total manifest rows = {len(rows)}")

    by_hash = defaultdict(list)

    for i, r in enumerate(rows, 1):
        if i % 200 == 0:
            print(f"[INFO] hashing {i}/{len(rows)}")
        h = md5_file(r["path"])
        by_hash[h].append(r)

    within_split = []
    cross_split = []

    for h, rs in by_hash.items():
        if len(rs) <= 1:
            continue

        splits = sorted(set(r["split"] for r in rs))
        if len(splits) == 1:
            within_split.append((h, rs))
        else:
            cross_split.append((h, rs))

    print("\n[HASH DUPLICATE OVERLAP]")
    print(f"within-split duplicate groups = {len(within_split)}")
    print(f"cross-split duplicate groups  = {len(cross_split)}")

    if cross_split:
        print("\n[CROSS-SPLIT EXAMPLES]")
        for h, rs in cross_split[:20]:
            print(f"\nMD5 = {h}")
            for r in rs:
                print(f"  {r['split']:5s} {r['rel']}")

    if len(cross_split) > 0:
        raise SystemExit("[FAIL] Hash duplicates exist across train/val/test.")
    else:
        print("[OK] No hash duplicate leakage across train/val/test.")


if __name__ == "__main__":
    main()