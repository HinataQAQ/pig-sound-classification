from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(r"C:\py\pigsound\pig-sound-classification")

IN_MANIFESTS = [
    ROOT / "data" / "manifests_cough_silence" / "train.csv",
    ROOT / "data" / "manifests_cough_silence" / "val.csv",
    ROOT / "data" / "manifests_cough_silence" / "test.csv",
]

OUT_DIR = ROOT / "data" / "manifests_source_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED_BASE = 3407
N_TRIES = 500

TEST_SIZE = 0.20
VAL_SIZE = 0.20


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


def source_id_from_path(p: str) -> str:
    """
    从 1 秒切片文件名提取来源长录音 ID。
    例：
    aswine_ala_e_2_2020-09-30_05-58-11_488_t0000053000.wav
    -> aswine_ala_e_2_2020-09-30_05-58-11_488
    """
    stem = Path(str(p)).stem.lower()

    m = re.match(r"(.+?)_t\d+$", stem)
    if m:
        return m.group(1)

    # 兼容 event/topk clip 命名
    if "__rank" in stem:
        return stem.split("__rank")[0]

    m = re.match(r"(.+?)_rank\d+", stem)
    if m:
        return m.group(1)

    return stem


def load_all():
    rows = []

    for mp in IN_MANIFESTS:
        df = pd.read_csv(mp)
        path_col = infer_path_col(df)
        label_col = infer_label_col(df)

        for _, r in df.iterrows():
            fp = str(r[path_col])
            lab = str(r[label_col]).strip().lower()
            rows.append({
                "filepath": fp,
                "label": lab,
                "source_id": source_id_from_path(fp),
                "origin_manifest": str(mp),
            })

    all_df = pd.DataFrame(rows)

    # 去掉完全重复路径
    all_df["_norm_path"] = all_df["filepath"].map(lambda x: str(Path(str(x)).resolve()).lower())
    all_df = all_df.drop_duplicates(subset=["_norm_path"]).drop(columns=["_norm_path"]).copy()

    return all_df


def label_ratio(df):
    vc = df["label"].value_counts(normalize=True)
    return float(vc.get("cough", 0.0))


def score_split(train_df, val_df, test_df, target_ratio):
    """
    越小越好：
    1. cough/other 比例接近总体比例
    2. split size 接近 60/20/20
    """
    n = len(train_df) + len(val_df) + len(test_df)

    r_tr = label_ratio(train_df)
    r_va = label_ratio(val_df)
    r_te = label_ratio(test_df)

    size_tr = len(train_df) / n
    size_va = len(val_df) / n
    size_te = len(test_df) / n

    score = 0.0
    score += abs(r_tr - target_ratio)
    score += abs(r_va - target_ratio)
    score += abs(r_te - target_ratio)

    score += abs(size_tr - 0.60)
    score += abs(size_va - 0.20)
    score += abs(size_te - 0.20)

    return score


def make_split(df, seed):
    groups = df["source_id"].values

    # first split: trainval vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # second split: train vs val, val should be 20% total = 25% of trainval
    val_frac_of_trainval = VAL_SIZE / (1.0 - TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed + 10000)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df["source_id"].values))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def check_overlap(train_df, val_df, test_df):
    tr = set(train_df["source_id"])
    va = set(val_df["source_id"])
    te = set(test_df["source_id"])

    print("[SOURCE OVERLAP]")
    print("train-val :", len(tr & va))
    print("train-test:", len(tr & te))
    print("val-test  :", len(va & te))

    ptr = set(train_df["filepath"].map(lambda x: str(Path(str(x)).resolve()).lower()))
    pva = set(val_df["filepath"].map(lambda x: str(Path(str(x)).resolve()).lower()))
    pte = set(test_df["filepath"].map(lambda x: str(Path(str(x)).resolve()).lower()))

    print("[EXACT PATH OVERLAP]")
    print("train-val :", len(ptr & pva))
    print("train-test:", len(ptr & pte))
    print("val-test  :", len(pva & pte))


def write_split(name, df):
    out = OUT_DIR / f"{name}.csv"
    df[["filepath", "label"]].to_csv(out, index=False, encoding="utf-8-sig")

    full = OUT_DIR / f"{name}_full.csv"
    df.to_csv(full, index=False, encoding="utf-8-sig")

    print(f"[OK] wrote {name}: {len(df)} -> {out}")
    print(df["label"].value_counts())


def main():
    df = load_all()
    print("[INFO] total rows =", len(df))
    print("[INFO] total sources =", df["source_id"].nunique())
    print("[INFO] overall label counts:")
    print(df["label"].value_counts())

    target_ratio = label_ratio(df)

    best = None
    best_score = 1e9

    for i in range(N_TRIES):
        seed = SEED_BASE + i
        tr, va, te = make_split(df, seed)
        s = score_split(tr, va, te, target_ratio)

        if s < best_score:
            best_score = s
            best = (seed, tr, va, te)

    seed, train_df, val_df, test_df = best

    print("\n[INFO] best seed =", seed)
    print("[INFO] best score =", best_score)

    check_overlap(train_df, val_df, test_df)

    write_split("train", train_df)
    write_split("val", val_df)
    write_split("test", test_df)


if __name__ == "__main__":
    main()