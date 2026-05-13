import argparse
from pathlib import Path

import pandas as pd


def gather_wavs(folder: Path, label: str):
    if folder is None or (not folder.exists()):
        return []
    rows = []
    for p in sorted(folder.glob("*.wav")):
        rows.append({"filepath": str(p.resolve()), "label": label})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="包含4个子文件夹的根目录")
    ap.add_argument("--train_manifest", required=True, help="原始训练集 manifest")
    ap.add_argument("--out_dir", default="data/manifests_round1_review", help="输出目录")

    # 如果你的文件夹名字不同，就在命令里改这几个参数
    ap.add_argument("--other_dir", default="other_clean")
    ap.add_argument("--cough_dir", default="cough_clean")
    ap.add_argument("--mixed_dir", default="cough_mixed")
    ap.add_argument("--uncertain_dir", default="uncertain_hold")

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    other_folder = base_dir / args.other_dir
    cough_folder = base_dir / args.cough_dir
    mixed_folder = base_dir / args.mixed_dir
    uncertain_folder = base_dir / args.uncertain_dir

    # 分别收集
    other_rows = gather_wavs(other_folder, "other")
    cough_rows = gather_wavs(cough_folder, "cough")
    mixed_rows = gather_wavs(mixed_folder, "cough")
    uncertain_rows = gather_wavs(uncertain_folder, "other")

    df_other = pd.DataFrame(other_rows)
    df_cough = pd.DataFrame(cough_rows)
    df_mixed = pd.DataFrame(mixed_rows)
    df_uncertain = pd.DataFrame(uncertain_rows)

    # 分开存档
    other_csv = out_dir / "round1_other_clean.csv"
    cough_csv = out_dir / "round1_cough_clean.csv"
    mixed_csv = out_dir / "round1_cough_mixed_hold.csv"
    uncertain_csv = out_dir / "round1_uncertain_hold.csv"

    df_other.to_csv(other_csv, index=False, encoding="utf-8-sig")
    df_cough.to_csv(cough_csv, index=False, encoding="utf-8-sig")
    df_mixed.to_csv(mixed_csv, index=False, encoding="utf-8-sig")
    df_uncertain.to_csv(uncertain_csv, index=False, encoding="utf-8-sig")

    # 读取原 train manifest
    train_df = pd.read_csv(args.train_manifest)

    # round1 只先回灌 other_clean
    train_plus_other = pd.concat([train_df, df_other], ignore_index=True)
    train_plus_other = train_plus_other.sample(frac=1.0, random_state=3407).reset_index(drop=True)

    train_plus_other_csv = out_dir / "train_with_round1_other_only.csv"
    train_plus_other.to_csv(train_plus_other_csv, index=False, encoding="utf-8-sig")

    # 可选：同时把 cough_clean 也回灌一版，先存出来，但你暂时别急着用
    if len(df_cough) > 0:
        train_plus_both = pd.concat([train_df, df_other, df_cough], ignore_index=True)
        train_plus_both = train_plus_both.sample(frac=1.0, random_state=3407).reset_index(drop=True)
        train_plus_both_csv = out_dir / "train_with_round1_other_and_cough_clean.csv"
        train_plus_both.to_csv(train_plus_both_csv, index=False, encoding="utf-8-sig")
    else:
        train_plus_both_csv = None

    print("[OK] saved:")
    print("  other_clean   ->", other_csv.resolve())
    print("  cough_clean   ->", cough_csv.resolve())
    print("  cough_mixed   ->", mixed_csv.resolve())
    print("  uncertain     ->", uncertain_csv.resolve())
    print("  train+other   ->", train_plus_other_csv.resolve())
    if train_plus_both_csv is not None:
        print("  train+both    ->", train_plus_both_csv.resolve())

    print("\n[COUNT]")
    print("  other_clean =", len(df_other))
    print("  cough_clean =", len(df_cough))
    print("  cough_mixed =", len(df_mixed))
    print("  uncertain   =", len(df_uncertain))
    print("  original_train =", len(train_df))
    print("  train_with_round1_other_only =", len(train_plus_other))


if __name__ == "__main__":
    main()