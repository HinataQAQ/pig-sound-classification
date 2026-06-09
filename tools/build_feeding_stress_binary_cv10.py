from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def infer_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column from columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_cv10",
        type=str,
        default="data/manifests_pigvocal_4class_dedup_cv10",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="data/manifests_pigvocal_feeding_stress_cv10",
    )
    args = ap.parse_args()

    base = ROOT / args.base_cv10
    out_root = ROOT / args.out_root

    keep_labels = {"feeding", "stress_vocal"}

    for fold in range(10):
        print(f"\n===== fold {fold} =====")

        in_dir = base / f"fold{fold}"
        out_dir = out_root / f"fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            src = in_dir / f"{split}.csv"
            df = pd.read_csv(src)

            label_col = infer_label_col(df)
            df[label_col] = df[label_col].astype(str).str.strip().str.lower()

            sub = df[df[label_col].isin(keep_labels)].copy()
            sub = sub.reset_index(drop=True)

            out = out_dir / f"{split}.csv"
            sub.to_csv(out, index=False, encoding="utf-8-sig")

            print(f"{split:5s}: {len(sub):4d}  {dict(sub[label_col].value_counts())}")

    print(f"\n[OK] wrote -> {out_root}")


if __name__ == "__main__":
    main()