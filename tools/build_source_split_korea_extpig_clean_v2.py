from pathlib import Path
import argparse
import random
import re
import pandas as pd
import librosa


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


def is_valid_audio(path: Path, sr=32000) -> bool:
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=0.2)
        return y is not None and len(y) > 0
    except Exception:
        return False


def sample_valid(files, n, rng, tag):
    files = list(files)
    rng.shuffle(files)

    kept = []
    bad = []

    for p in files:
        if len(kept) >= n:
            break

        if is_valid_audio(p):
            kept.append(p)
        else:
            bad.append(p)

    print(f"[INFO] {tag}: requested={n}, kept={len(kept)}, bad_checked={len(bad)}")
    return kept, bad


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


def add_rows(rows, files, label, source, repeat=1):
    for p in files:
        for _ in range(max(1, repeat)):
            rows.append({
                "filepath": str(p.resolve()),
                "label": label,
                "source": source,
            })


def add_clean_cough_replay(rows, folders, n, repeat, rng, forbidden_sources):
    all_files = []
    for d in folders:
        all_files.extend(collect_audio(Path(d)))

    kept = []
    excluded = []
    bad = []

    rng.shuffle(all_files)

    for p in all_files:
        if len(kept) >= n:
            break

        sid = canonical_source_id(str(p))
        if sid in forbidden_sources:
            excluded.append(p)
            continue

        if is_valid_audio(p):
            kept.append(p)
        else:
            bad.append(p)

    add_rows(rows, kept, "cough", "local_clean_cough_replay", repeat=repeat)

    print(f"[INFO] local clean cough found={len(all_files)} kept={len(kept)} excluded_source={len(excluded)} bad={len(bad)} repeat={repeat}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--val_full", required=True)
    ap.add_argument("--test_full", required=True)

    ap.add_argument("--korea_dry_dir", required=True)
    ap.add_argument("--korea_abdominal_dir", required=True)
    ap.add_argument("--soundwel_dir", required=True)
    ap.add_argument("--wu_dir", required=True)
    ap.add_argument("--clean_cough_dirs", nargs="+", required=True)

    ap.add_argument("--n_korea_dry", type=int, default=1000)
    ap.add_argument("--n_korea_abdominal", type=int, default=1000)
    ap.add_argument("--n_soundwel", type=int, default=1000)
    ap.add_argument("--n_wu", type=int, default=1000)
    ap.add_argument("--n_clean_cough", type=int, default=200)
    ap.add_argument("--repeat_clean_cough", type=int, default=2)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_full_csv", required=True)
    ap.add_argument("--bad_csv", required=True)

    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = load_base_train(Path(args.base_train))
    forbidden_sources = load_forbidden_sources(Path(args.val_full), Path(args.test_full))

    print(f"[INFO] base train rows = {len(rows)}")
    print(f"[INFO] forbidden val/test source ids = {len(forbidden_sources)}")

    korea_dry = collect_audio(Path(args.korea_dry_dir))
    korea_abd = collect_audio(Path(args.korea_abdominal_dir))
    soundwel = collect_audio(Path(args.soundwel_dir))
    wu = collect_audio(Path(args.wu_dir))

    print(f"[INFO] korea dry total = {len(korea_dry)}")
    print(f"[INFO] korea abdominal total = {len(korea_abd)}")
    print(f"[INFO] soundwel total = {len(soundwel)}")
    print(f"[INFO] wu total = {len(wu)}")

    dry_keep, dry_bad = sample_valid(korea_dry, args.n_korea_dry, rng, "korea_dry_cough")
    abd_keep, abd_bad = sample_valid(korea_abd, args.n_korea_abdominal, rng, "korea_abdominal_cough")
    soundwel_keep, soundwel_bad = sample_valid(soundwel, args.n_soundwel, rng, "soundwel_other")
    wu_keep, wu_bad = sample_valid(wu, args.n_wu, rng, "wu_other")

    add_rows(rows, dry_keep, "cough", "external_cough_korea_dry")
    add_rows(rows, abd_keep, "cough", "external_cough_korea_abdominal")
    add_rows(rows, soundwel_keep, "other", "external_other_soundwel_pig_vocal")
    add_rows(rows, wu_keep, "other", "external_other_wu_pig_speech")

    clean_bad = add_clean_cough_replay(
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

    bad_rows = []
    for p in dry_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "external_cough_korea_dry", "reason": "invalid_audio"})
    for p in abd_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "external_cough_korea_abdominal", "reason": "invalid_audio"})
    for p in soundwel_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "external_other_soundwel_pig_vocal", "reason": "invalid_audio"})
    for p in wu_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "external_other_wu_pig_speech", "reason": "invalid_audio"})
    for p in clean_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "local_clean_cough_replay", "reason": "invalid_audio"})
    for _, r in missing_df.iterrows():
        bad_rows.append({"filepath": str(r["filepath"]), "source": str(r.get("source", "")), "reason": "missing_file"})

    out_csv = Path(args.out_csv)
    out_full = Path(args.out_full_csv)
    bad_csv = Path(args.bad_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_full, index=False, encoding="utf-8-sig")
    full_df[["filepath", "label"]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(bad_rows).to_csv(bad_csv, index=False, encoding="utf-8-sig")

    print("[OK] wrote train ->", out_csv)
    print("[OK] wrote full  ->", out_full)
    print("[OK] wrote bad   ->", bad_csv)

    print("\n[INFO] label counts:")
    print(full_df["label"].value_counts())

    print("\n[INFO] source counts:")
    print(full_df["source"].value_counts())


if __name__ == "__main__":
    main()