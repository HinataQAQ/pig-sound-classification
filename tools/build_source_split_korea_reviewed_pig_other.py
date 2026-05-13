from pathlib import Path
import argparse
import random
import pandas as pd
import librosa


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def infer_path_col(df):
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column. columns={list(df.columns)}")


def infer_label_col(df):
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column. columns={list(df.columns)}")


def collect_audio(folder: Path):
    if not folder.exists():
        print("[WARN] missing:", folder)
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def is_valid_audio(path: Path, sr=32000):
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
        if n > 0 and len(kept) >= n:
            break
        if is_valid_audio(p):
            kept.append(p)
        else:
            bad.append(p)

    print(f"[INFO] {tag}: requested={n}, kept={len(kept)}, bad={len(bad)}")
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


def add_rows(rows, files, label, source, repeat=1):
    for p in files:
        for _ in range(max(1, repeat)):
            rows.append({
                "filepath": str(p.resolve()),
                "label": label,
                "source": source,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--korea_dry_dir", required=True)
    ap.add_argument("--korea_abdominal_dir", required=True)
    ap.add_argument("--review_root", required=True)

    ap.add_argument("--n_korea_dry", type=int, default=500)
    ap.add_argument("--n_korea_abdominal", type=int, default=500)
    ap.add_argument("--repeat_reviewed_other", type=int, default=3)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_full_csv", required=True)
    ap.add_argument("--bad_csv", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_base_train(Path(args.base_train))

    korea_dry = collect_audio(Path(args.korea_dry_dir))
    korea_abd = collect_audio(Path(args.korea_abdominal_dir))

    dry_keep, dry_bad = sample_valid(korea_dry, args.n_korea_dry, rng, "korea_dry")
    abd_keep, abd_bad = sample_valid(korea_abd, args.n_korea_abdominal, rng, "korea_abdominal")

    add_rows(rows, dry_keep, "cough", "external_cough_korea_dry")
    add_rows(rows, abd_keep, "cough", "external_cough_korea_abdominal")

    review_root = Path(args.review_root)

    reviewed_dirs = [
        ("pig_grunt_oink", "reviewed_other_pig_grunt_oink"),
        ("hard_other_clean", "reviewed_other_hard_other_clean"),
        ("pig_squeal_whistle", "reviewed_other_pig_squeal_whistle"),
    ]

    for subdir, source_name in reviewed_dirs:
        files = collect_audio(review_root / subdir)
        add_rows(
            rows,
            files,
            label="other",
            source=source_name,
            repeat=args.repeat_reviewed_other,
        )
        print(f"[INFO] {subdir}: files={len(files)}, rows={len(files) * args.repeat_reviewed_other}")

    full_df = pd.DataFrame(rows)

    exists = full_df["filepath"].apply(lambda x: Path(str(x)).exists())
    missing = full_df[~exists].copy()
    full_df = full_df[exists].copy()

    bad_rows = []
    for p in dry_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "external_cough_korea_dry", "reason": "invalid_audio"})
    for p in abd_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "external_cough_korea_abdominal", "reason": "invalid_audio"})
    for _, r in missing.iterrows():
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