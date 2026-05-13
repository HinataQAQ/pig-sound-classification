from pathlib import Path
import argparse
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


def collect_audio(folder: Path):
    if not folder.exists():
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def collect_reviewed(review_root: Path):
    """
    扫描 round4 人工复核目录：
      reviewed/cough_clean -> cough
      reviewed/other_clean/** -> other
    uncertain 不进入训练。
    """
    rows = []

    cough_dir = review_root / "cough_clean"
    for p in collect_audio(cough_dir):
        rows.append({
            "filepath": str(p.resolve()),
            "label": "cough",
            "source": "round4_review_cough_clean",
        })

    other_dir = review_root / "other_clean"
    for p in collect_audio(other_dir):
        subtype = "unknown"
        try:
            rel = p.relative_to(other_dir)
            if len(rel.parts) > 1:
                subtype = rel.parts[0]
        except Exception:
            pass

        rows.append({
            "filepath": str(p.resolve()),
            "label": "other",
            "source": f"round4_review_other_clean/{subtype}",
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_manifest", required=True)
    ap.add_argument("--review_root", required=True)
    ap.add_argument("--aug_manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_full_csv", required=True)
    ap.add_argument("--repeat_round4_other", type=int, default=2)
    ap.add_argument("--repeat_aug_cough", type=int, default=1)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_full_csv = Path(args.out_full_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. 原始 rebuilt 主训练数据
    base_df = pd.read_csv(args.base_manifest)
    base_path_col = infer_path_col(base_df)
    base_label_col = infer_label_col(base_df)

    base_rows = []
    for _, r in base_df.iterrows():
        base_rows.append({
            "filepath": str(r[base_path_col]),
            "label": str(r[base_label_col]).strip().lower(),
            "source": "base_rebuilt",
        })

    # 2. round4 人工复核 clean 数据
    review_rows = collect_reviewed(Path(args.review_root))

    # 3. round4 混噪增强 cough
    aug_df = pd.read_csv(args.aug_manifest)
    aug_path_col = infer_path_col(aug_df)
    aug_label_col = infer_label_col(aug_df)

    aug_rows = []
    for _, r in aug_df.iterrows():
        aug_rows.append({
            "filepath": str(r[aug_path_col]),
            "label": str(r[aug_label_col]).strip().lower(),
            "source": "round4_noise_mixed_cough",
        })

    # 4. 对 round4 hard negatives 加权：重复写入 manifest
    expanded_review_rows = []
    for r in review_rows:
        if r["label"] == "other":
            for _ in range(max(1, args.repeat_round4_other)):
                expanded_review_rows.append(dict(r))
        else:
            expanded_review_rows.append(dict(r))

    # 5. 混噪 cough 也可以重复，这里默认 1 次
    expanded_aug_rows = []
    for r in aug_rows:
        for _ in range(max(1, args.repeat_aug_cough)):
            expanded_aug_rows.append(dict(r))

    all_rows = base_rows + expanded_review_rows + expanded_aug_rows
    full_df = pd.DataFrame(all_rows)

    # 6. 过滤不存在的路径，防止训练中断
    exists_mask = full_df["filepath"].apply(lambda x: Path(str(x)).exists())
    missing_df = full_df[~exists_mask].copy()
    full_df = full_df[exists_mask].copy()

    # 7. 训练实际只需要 filepath,label 两列
    train_df = full_df[["filepath", "label"]].copy()

    full_df.to_csv(out_full_csv, index=False, encoding="utf-8-sig")
    train_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[OK] wrote train ->", out_csv)
    print("[OK] wrote full  ->", out_full_csv)
    print("[INFO] missing rows =", len(missing_df))

    print("\n[INFO] label counts:")
    print(train_df["label"].value_counts())

    print("\n[INFO] source counts:")
    print(full_df["source"].value_counts())


if __name__ == "__main__":
    main()