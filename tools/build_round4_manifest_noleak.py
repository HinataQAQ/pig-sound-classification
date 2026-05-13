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


def norm_path(p) -> str:
    return str(Path(str(p)).resolve()).lower()


def collect_audio(folder: Path):
    if not folder.exists():
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def collect_reviewed(review_root: Path):
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


def load_manifest_rows(path: Path, source_name: str):
    df = pd.read_csv(path)
    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "filepath": str(r[path_col]),
            "label": str(r[label_col]).strip().lower(),
            "source": source_name,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
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

    # 1. 读取 val/test，构建禁止进入训练的路径集合
    val_df = pd.read_csv(args.val_manifest)
    test_df = pd.read_csv(args.test_manifest)
    val_path_col = infer_path_col(val_df)
    test_path_col = infer_path_col(test_df)

    forbidden_paths = set(val_df[val_path_col].map(norm_path)) | set(test_df[test_path_col].map(norm_path))
    print("[INFO] forbidden val/test paths =", len(forbidden_paths))

    # 2. 基础训练集：必须用原始 train.csv，不能再用 train_rebuilt.csv
    base_rows = load_manifest_rows(Path(args.base_train), "base_train_original")

    # 3. round4 人工复核数据
    review_rows = collect_reviewed(Path(args.review_root))

    # 4. 增强 cough
    aug_df = pd.read_csv(args.aug_manifest)
    aug_path_col = infer_path_col(aug_df)
    aug_label_col = infer_label_col(aug_df)

    aug_rows = []
    excluded_aug_source_overlap = 0

    for _, r in aug_df.iterrows():
        fp = str(r[aug_path_col])
        lab = str(r[aug_label_col]).strip().lower()

        # 如果增强样本的原始 cough_source/noise_source 恰好来自 val/test，排除
        bad_source = False
        for c in ["cough_source", "noise_source"]:
            if c in aug_df.columns:
                src_path = str(r[c])
                if norm_path(src_path) in forbidden_paths:
                    bad_source = True

        if bad_source:
            excluded_aug_source_overlap += 1
            continue

        aug_rows.append({
            "filepath": fp,
            "label": lab,
            "source": "round4_noise_mixed_cough",
        })

    # 5. hard negative 重复写入，提高采样权重
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

    # 6. 过滤掉不存在文件
    exists_mask = full_df["filepath"].apply(lambda x: Path(str(x)).exists())
    missing_df = full_df[~exists_mask].copy()
    full_df = full_df[exists_mask].copy()

    # 7. 过滤掉和 val/test 完全相同的路径
    before = len(full_df)
    full_df["_norm_path"] = full_df["filepath"].map(norm_path)
    full_df = full_df[~full_df["_norm_path"].isin(forbidden_paths)].copy()
    exact_excluded = before - len(full_df)
    full_df = full_df.drop(columns=["_norm_path"])

    # 8. 输出
    train_df = full_df[["filepath", "label"]].copy()

    full_df.to_csv(out_full_csv, index=False, encoding="utf-8-sig")
    train_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[OK] wrote train ->", out_csv)
    print("[OK] wrote full  ->", out_full_csv)
    print("[INFO] missing rows =", len(missing_df))
    print("[INFO] exact val/test excluded =", exact_excluded)
    print("[INFO] augmented source-overlap excluded =", excluded_aug_source_overlap)

    print("\n[INFO] label counts:")
    print(train_df["label"].value_counts())

    print("\n[INFO] source counts:")
    print(full_df["source"].value_counts())


if __name__ == "__main__":
    main()