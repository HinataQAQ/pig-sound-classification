from pathlib import Path
import pandas as pd
import re


ROOT = Path(r"C:\py\pigsound\pig-sound-classification")

TRAIN_FULL = ROOT / "data" / "manifests_round4" / "train_full.csv"
TRAIN_CSV = ROOT / "data" / "manifests_round4" / "train.csv"
TEST_CSV = ROOT / "data" / "manifests_cough_silence" / "test.csv"
AUG_CSV = ROOT / "data" / "augmented" / "round4_noise_mix" / "manifest.csv"


def infer_path_col(df):
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot find path column in columns={list(df.columns)}")


def normalize_path(p):
    return str(Path(str(p)).resolve()).lower()


def rough_source_id(path_str):
    """
    尽量从文件名里提取来源长录音/来源片段标识。
    这个不是精确泄漏检测，只是帮助发现同源风险。
    """
    stem = Path(path_str).stem

    # round4 导出的事件切片常见格式：
    # ALA_E_...__rank001__p0.900__123.00-124.00
    if "__rank" in stem:
        return stem.split("__rank")[0]

    # 旧 TopK 切片常见格式：
    # ALA_000347_rank027_p0.820_src...
    m = re.match(r"(.+?)_rank\d+", stem)
    if m:
        return m.group(1)

    # 增强文件可能是 mix_00000__snr...，不能直接代表来源
    if stem.startswith("mix_"):
        return "AUGMENTED_MIX"

    return stem


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_path_col = infer_path_col(train_df)
    test_path_col = infer_path_col(test_df)

    train_paths = set(train_df[train_path_col].map(normalize_path))
    test_paths = set(test_df[test_path_col].map(normalize_path))

    exact_overlap = train_paths & test_paths

    print("[EXACT PATH OVERLAP]")
    print("count =", len(exact_overlap))
    for p in list(exact_overlap)[:20]:
        print(p)

    train_ids = set(train_df[train_path_col].map(lambda x: rough_source_id(str(x))))
    test_ids = set(test_df[test_path_col].map(lambda x: rough_source_id(str(x))))

    source_overlap = train_ids & test_ids

    print("\n[ROUGH SOURCE ID OVERLAP]")
    print("count =", len(source_overlap))
    for s in sorted(list(source_overlap))[:50]:
        print(s)

    if TRAIN_FULL.exists():
        full_df = pd.read_csv(TRAIN_FULL)
        if "source" in full_df.columns:
            print("\n[TRAIN SOURCE COUNTS]")
            print(full_df["source"].value_counts())

    if AUG_CSV.exists():
        aug_df = pd.read_csv(AUG_CSV)
        print("\n[AUGMENTED SOURCE CHECK]")
        for c in ["cough_source", "noise_source"]:
            if c in aug_df.columns:
                aug_sources = set(aug_df[c].map(normalize_path))
                aug_test_overlap = aug_sources & test_paths
                print(f"{c} exact overlap with test:", len(aug_test_overlap))
                for p in list(aug_test_overlap)[:20]:
                    print(p)


if __name__ == "__main__":
    main()