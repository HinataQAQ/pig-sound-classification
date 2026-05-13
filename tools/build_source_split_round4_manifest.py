from pathlib import Path
import argparse
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
    """
    把不同命名方式统一成 source id。
    例：
    aswine_ALA_E_2_2020-09-30_00-21-48_316_t0000009000
    -> ala_e_2_2020-09-30_00-21-48_316

    ALA_E_2_2020-09-30_00-21-48_316__rank001...
    -> ala_e_2_2020-09-30_00-21-48_316
    """
    stem = Path(str(path_or_name)).stem.lower()

    if stem.startswith("aswine_"):
        stem = stem[len("aswine_"):]

    # processed 1s slices: xxx_t0000123000
    m = re.match(r"(.+?)_t\d+$", stem)
    if m:
        return m.group(1)

    # event clips: xxx__rank001__p...
    if "__rank" in stem:
        return stem.split("__rank")[0]

    # old clips: xxx_rank001_p...
    m = re.match(r"(.+?)_rank\d+", stem)
    if m:
        return m.group(1)

    # mixed augmented wav: mix_00000__snr...
    if stem.startswith("mix_"):
        return "augmented_mix"

    return stem


def collect_audio(folder: Path):
    if not folder.exists():
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def load_manifest_rows(csv_path: Path, source_name: str):
    df = pd.read_csv(csv_path)
    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    rows = []
    for _, r in df.iterrows():
        fp = str(r[path_col])
        lab = str(r[label_col]).strip().lower()
        rows.append({
            "filepath": fp,
            "label": lab,
            "source": source_name,
            "source_id": canonical_source_id(fp),
        })
    return rows


def load_source_ids(full_csv: Path):
    df = pd.read_csv(full_csv)

    if "source_id" in df.columns:
        ids = set(str(x).strip().lower() for x in df["source_id"])
        # 同时兼容去掉 aswine_ 前缀后的形式
        ids2 = set()
        for x in ids:
            if x.startswith("aswine_"):
                ids2.add(x[len("aswine_"):])
            ids2.add(x)
        return ids2

    path_col = infer_path_col(df)
    return set(canonical_source_id(x) for x in df[path_col])


def collect_reviewed(review_root: Path, forbidden_source_ids: set):
    rows = []
    excluded = []

    # cough_clean
    cough_dir = review_root / "cough_clean"
    for p in collect_audio(cough_dir):
        sid = canonical_source_id(str(p))
        row = {
            "filepath": str(p.resolve()),
            "label": "cough",
            "source": "round4_review_cough_clean",
            "source_id": sid,
        }
        if sid in forbidden_source_ids:
            excluded.append(row)
        else:
            rows.append(row)

    # other_clean/*
    other_dir = review_root / "other_clean"
    for p in collect_audio(other_dir):
        sid = canonical_source_id(str(p))
        subtype = "unknown"
        try:
            rel = p.relative_to(other_dir)
            if len(rel.parts) > 1:
                subtype = rel.parts[0]
        except Exception:
            pass

        row = {
            "filepath": str(p.resolve()),
            "label": "other",
            "source": f"round4_review_other_clean/{subtype}",
            "source_id": sid,
        }
        if sid in forbidden_source_ids:
            excluded.append(row)
        else:
            rows.append(row)

    return rows, excluded


def load_aug_rows(aug_manifest: Path, forbidden_source_ids: set):
    df = pd.read_csv(aug_manifest)
    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    rows = []
    excluded = []

    for _, r in df.iterrows():
        fp = str(r[path_col])
        lab = str(r[label_col]).strip().lower()

        bad = False
        bad_reason = []

        for c in ["cough_source", "noise_source"]:
            if c in df.columns:
                sid = canonical_source_id(str(r[c]))
                if sid in forbidden_source_ids:
                    bad = True
                    bad_reason.append(f"{c}:{sid}")

        row = {
            "filepath": fp,
            "label": lab,
            "source": "round4_noise_mixed_cough",
            "source_id": "augmented_mix",
        }

        if bad:
            row["exclude_reason"] = ";".join(bad_reason)
            excluded.append(row)
        else:
            rows.append(row)

    return rows, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_train", required=True)
    ap.add_argument("--source_train_full", required=True)
    ap.add_argument("--source_val_full", required=True)
    ap.add_argument("--source_test_full", required=True)
    ap.add_argument("--review_root", required=True)
    ap.add_argument("--aug_manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_full_csv", required=True)
    ap.add_argument("--out_excluded_csv", required=True)
    ap.add_argument("--repeat_round4_other", type=int, default=1)
    ap.add_argument("--repeat_aug_cough", type=int, default=1)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_full_csv = Path(args.out_full_csv)
    out_excluded_csv = Path(args.out_excluded_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    val_ids = load_source_ids(Path(args.source_val_full))
    test_ids = load_source_ids(Path(args.source_test_full))
    forbidden_source_ids = val_ids | test_ids

    print("[INFO] forbidden val/test source ids =", len(forbidden_source_ids))

    base_rows = load_manifest_rows(Path(args.source_train), "source_split_train_original")

    review_rows, review_excluded = collect_reviewed(
        Path(args.review_root),
        forbidden_source_ids=forbidden_source_ids,
    )

    aug_rows, aug_excluded = load_aug_rows(
        Path(args.aug_manifest),
        forbidden_source_ids=forbidden_source_ids,
    )

    expanded_review_rows = []
    for r in review_rows:
        if r["label"] == "other":
            for _ in range(max(1, args.repeat_round4_other)):
                expanded_review_rows.append(dict(r))
        else:
            expanded_review_rows.append(dict(r))

    expanded_aug_rows = []
    for r in aug_rows:
        for _ in range(max(1, args.repeat_aug_cough)):
            expanded_aug_rows.append(dict(r))

    all_rows = base_rows + expanded_review_rows + expanded_aug_rows
    full_df = pd.DataFrame(all_rows)

    # 路径存在性过滤
    exists_mask = full_df["filepath"].apply(lambda x: Path(str(x)).exists())
    missing_df = full_df[~exists_mask].copy()
    full_df = full_df[exists_mask].copy()

    train_df = full_df[["filepath", "label"]].copy()

    excluded_df = pd.DataFrame(review_excluded + aug_excluded)

    full_df.to_csv(out_full_csv, index=False, encoding="utf-8-sig")
    train_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    excluded_df.to_csv(out_excluded_csv, index=False, encoding="utf-8-sig")

    print("[OK] wrote train ->", out_csv)
    print("[OK] wrote full  ->", out_full_csv)
    print("[OK] wrote excluded ->", out_excluded_csv)

    print("[INFO] missing rows =", len(missing_df))
    print("[INFO] review kept =", len(review_rows), "excluded =", len(review_excluded))
    print("[INFO] aug kept =", len(aug_rows), "excluded =", len(aug_excluded))

    print("\n[INFO] label counts:")
    print(train_df["label"].value_counts())

    print("\n[INFO] source counts:")
    print(full_df["source"].value_counts())


if __name__ == "__main__":
    main()