from pathlib import Path
import argparse
import random
import re
import pandas as pd


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def infer_path_col(df: pd.DataFrame) -> str:
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column. columns={list(df.columns)}")


def infer_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column. columns={list(df.columns)}")


def canonical_source_id(path_or_name: str) -> str:
    stem = Path(str(path_or_name)).stem.lower()

    if stem.startswith("aswine_"):
        stem = stem[len("aswine_"):]

    m = re.match(r"(.+?)_t\d+$", stem)
    if m:
        return m.group(1)

    if "__rank" in stem:
        return stem.split("__rank")[0]

    m = re.match(r"(.+?)_rank\d+", stem)
    if m:
        return m.group(1)

    return stem


def collect_audio(folder: Path):
    if not folder.exists():
        print(f"[WARN] missing folder: {folder}")
        return []

    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def sample_files(files, n, rng):
    if n <= 0:
        return []
    if len(files) > n:
        return rng.sample(files, n)
    return files


def load_forbidden_sources(val_full: Path, test_full: Path):
    ids = set()

    for fp in [val_full, test_full]:
        df = pd.read_csv(fp)
        if "source_id" in df.columns:
            for x in df["source_id"]:
                s = str(x).strip().lower()
                ids.add(s)
                if s.startswith("aswine_"):
                    ids.add(s[len("aswine_"):])
        else:
            path_col = infer_path_col(df)
            for x in df[path_col]:
                ids.add(canonical_source_id(x))

    return ids


def load_base_train(path: Path):
    df = pd.read_csv(path)
    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "filepath": str(r[path_col]),
            "label": str(r[label_col]).strip().lower(),
            "source": "source_split_train_original",
        })

    return rows


def add_external_other(rows, folder: Path, source_name: str, n: int, rng):
    files = collect_audio(folder)
    files = sample_files(files, n, rng)

    for p in files:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "other",
            "source": source_name,
        })

    print(f"[INFO] {source_name}: used {len(files)} files")


def add_clean_cough(rows, folders, n: int, repeat: int, rng, forbidden_sources):
    files = []
    for d in folders:
        files.extend(collect_audio(Path(d)))

    kept = []
    excluded = []

    for p in files:
        sid = canonical_source_id(str(p))
        if sid in forbidden_sources:
            excluded.append(p)
        else:
            kept.append(p)

    kept = sample_files(kept, n, rng)

    for p in kept:
        for _ in range(max(1, repeat)):
            rows.append({
                "filepath": str(p.resolve()),
                "label": "cough",
                "source": "clean_cough_replay",
            })

    print(f"[INFO] clean_cough original files found = {len(files)}")
    print(f"[INFO] clean_cough kept = {len(kept)}")
    print(f"[INFO] clean_cough excluded by val/test source = {len(excluded)}")
    print(f"[INFO] clean_cough rows after repeat = {len(kept) * max(1, repeat)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--val_full", required=True)
    ap.add_argument("--test_full", required=True)

    ap.add_argument("--soundwel_dir", required=True)
    ap.add_argument("--wu_dir", required=True)
    ap.add_argument("--clean_cough_dirs", nargs="+", required=True)

    ap.add_argument("--n_soundwel", type=int, default=500)
    ap.add_argument("--n_wu", type=int, default=500)
    ap.add_argument("--n_clean_cough", type=int, default=300)
    ap.add_argument("--repeat_clean_cough", type=int, default=3)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_full_csv", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = load_base_train(Path(args.base_train))
    forbidden_sources = load_forbidden_sources(Path(args.val_full), Path(args.test_full))

    print(f"[INFO] base rows = {len(rows)}")
    print(f"[INFO] forbidden val/test source ids = {len(forbidden_sources)}")

    add_external_other(
        rows,
        Path(args.soundwel_dir),
        source_name="external_other_soundwel_pig_vocal",
        n=args.n_soundwel,
        rng=rng,
    )

    add_external_other(
        rows,
        Path(args.wu_dir),
        source_name="external_other_wu_pig_speech",
        n=args.n_wu,
        rng=rng,
    )

    add_clean_cough(
        rows,
        folders=args.clean_cough_dirs,
        n=args.n_clean_cough,
        repeat=args.repeat_clean_cough,
        rng=rng,
        forbidden_sources=forbidden_sources,
    )

    full_df = pd.DataFrame(rows)

    exists_mask = full_df["filepath"].apply(lambda x: Path(str(x)).exists())
    missing_df = full_df[~exists_mask].copy()
    full_df = full_df[exists_mask].copy()

    out_csv = Path(args.out_csv)
    out_full = Path(args.out_full_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_full, index=False, encoding="utf-8-sig")
    full_df[["filepath", "label"]].to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[OK] wrote train ->", out_csv)
    print("[OK] wrote full  ->", out_full)
    print("[INFO] missing rows =", len(missing_df))

    print("\n[INFO] label counts:")
    print(full_df["label"].value_counts())

    print("\n[INFO] source counts:")
    print(full_df["source"].value_counts())


if __name__ == "__main__":
    main()