from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(r"C:\py\pigsound\pig-sound-classification")

TRAIN_FULL = ROOT / "data" / "manifests_source_split" / "train_full.csv"
VAL_FULL = ROOT / "data" / "manifests_source_split" / "val_full.csv"
TEST_FULL = ROOT / "data" / "manifests_source_split" / "test_full.csv"

OUT_DIR = ROOT / "data" / "manifests_source_trainval_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_SIZE = 0.15
SEED_BASE = 3407
N_TRIES = 300


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


def ensure_basic_columns(df):
    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    if "source_id" not in df.columns:
        raise RuntimeError(
            "source_id column not found. Please use train_full.csv / val_full.csv from source split."
        )

    out = df[[path_col, label_col, "source_id"]].copy()
    out.columns = ["filepath", "label", "source_id"]
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out["source_id"] = out["source_id"].astype(str).str.strip().str.lower()
    return out


def label_ratio(df):
    vc = df["label"].value_counts(normalize=True)
    return float(vc.get("cough", 0.0))


def score_split(train_df, dev_df, target_ratio):
    r_tr = label_ratio(train_df)
    r_dev = label_ratio(dev_df)

    size_dev = len(dev_df) / (len(train_df) + len(dev_df))

    score = 0.0
    score += abs(r_tr - target_ratio)
    score += abs(r_dev - target_ratio)
    score += abs(size_dev - DEV_SIZE)

    return score


def main():
    train_df = ensure_basic_columns(pd.read_csv(TRAIN_FULL))
    val_df = ensure_basic_columns(pd.read_csv(VAL_FULL))
    test_df = ensure_basic_columns(pd.read_csv(TEST_FULL))

    all_df = pd.concat([train_df, val_df], ignore_index=True)

    # 去重
    all_df["_norm"] = all_df["filepath"].map(lambda x: str(Path(str(x)).resolve()).lower())
    all_df = all_df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"]).copy()

    # 检查 test source 不进入 trainval
    trainval_sources = set(all_df["source_id"])
    test_sources = set(test_df["source_id"])
    overlap = trainval_sources & test_sources
    print("[CHECK] trainval-test source overlap =", len(overlap))

    if len(overlap) > 0:
        print("[WARN] examples:")
        for x in list(overlap)[:20]:
            print(x)
        raise RuntimeError("trainval and test source overlap detected. Stop.")

    target_ratio = label_ratio(all_df)

    best = None
    best_score = 1e9

    for i in range(N_TRIES):
        seed = SEED_BASE + i
        gss = GroupShuffleSplit(n_splits=1, test_size=DEV_SIZE, random_state=seed)
        tr_idx, dev_idx = next(gss.split(all_df, groups=all_df["source_id"].values))

        tr = all_df.iloc[tr_idx].reset_index(drop=True)
        dev = all_df.iloc[dev_idx].reset_index(drop=True)

        s = score_split(tr, dev, target_ratio)

        if s < best_score:
            best_score = s
            best = (seed, tr, dev)

    seed, tr, dev = best

    print("[INFO] best seed =", seed)
    print("[INFO] best score =", best_score)

    tr_sources = set(tr["source_id"])
    dev_sources = set(dev["source_id"])
    te_sources = set(test_df["source_id"])

    print("[SOURCE OVERLAP]")
    print("train-dev :", len(tr_sources & dev_sources))
    print("train-test:", len(tr_sources & te_sources))
    print("dev-test  :", len(dev_sources & te_sources))

    print("[COUNTS]")
    print("train_new =", len(tr))
    print(tr["label"].value_counts())
    print("dev_new =", len(dev))
    print(dev["label"].value_counts())
    print("test =", len(test_df))
    print(test_df["label"].value_counts())

    tr[["filepath", "label"]].to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    dev[["filepath", "label"]].to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8-sig")

    tr.to_csv(OUT_DIR / "train_full.csv", index=False, encoding="utf-8-sig")
    dev.to_csv(OUT_DIR / "val_full.csv", index=False, encoding="utf-8-sig")

    print("[OK] wrote ->", OUT_DIR / "train.csv")
    print("[OK] wrote ->", OUT_DIR / "val.csv")


if __name__ == "__main__":
    main()