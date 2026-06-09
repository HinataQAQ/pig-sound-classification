from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "mctafd_ablation_summary.csv"

df = pd.read_csv(CSV)


def parse_fold(exp: str):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_model(exp: str):
    if "cv5_gated_boundary_" in exp and "_mctafd_no_se_" in exp:
        if "_w02_" in exp:
            return "mctafd_no_se_gated_boundary_w02"
        if "_w05_" in exp:
            return "mctafd_no_se_gated_boundary_w05"
        if "_w10_" in exp:
            return "mctafd_no_se_gated_boundary_w10"
        return "mctafd_no_se_gated_boundary"

    if "cv5_gated_" in exp and "_mctafd_no_se_" in exp:
        return "mctafd_no_se_gated"

    if "cv5_boundary_" in exp and "_logmel_" in exp:
        if "_w02_" in exp:
            return "logmel_boundary_w02"
        if "_w05_" in exp:
            return "logmel_boundary_w05"
        if "_w10_" in exp:
            return "logmel_boundary_w10"
        return "logmel_boundary"

    if "cv5_boundary_" in exp and "_mctafd_no_se_" in exp:
        if "_w02_" in exp:
            return "mctafd_no_se_boundary_w02"
        if "_w05_" in exp:
            return "mctafd_no_se_boundary_w05"
        if "_w10_" in exp:
            return "mctafd_no_se_boundary_w10"
        return "mctafd_no_se_boundary"

    if exp.startswith("cv5_fold") and "_logmel_" in exp:
        return "logmel"

    if exp.startswith("cv5_fold") and "_mctafd_no_se_" in exp:
        return "mctafd_no_se_direct"

    if exp.startswith("cv5_fold") and "_mctafd_se_" in exp:
        return "mctafd_se"

    return None


cv = df[df["experiment"].str.startswith("cv5_", na=False)].copy()
cv["fold"] = cv["experiment"].apply(parse_fold)
cv["model"] = cv["experiment"].apply(parse_model)

cv = cv[cv["model"].notna()].copy()

cols = [
    "experiment",
    "model",
    "fold",
    "seed",
    "feature_mode",
    "use_se",
    "n_mels",
    "fmax",
    "best_val_macro_f1",
    "test_acc",
    "test_macro_f1",
]

for c in cols:
    if c not in cv.columns:
        cv[c] = None

cv = cv[cols].sort_values(["model", "seed", "fold", "experiment"])

print("\n[CV5 VARIANT RUNS]")
print(cv.to_string(index=False))

summary = (
    cv.groupby("model")
      .agg(
          n=("test_macro_f1", "count"),
          mean_macro_f1=("test_macro_f1", "mean"),
          std_macro_f1=("test_macro_f1", "std"),
          min_macro_f1=("test_macro_f1", "min"),
          max_macro_f1=("test_macro_f1", "max"),
          mean_best_val=("best_val_macro_f1", "mean"),
      )
      .reset_index()
      .sort_values("mean_macro_f1", ascending=False)
)

print("\n[CV5 VARIANT SUMMARY]")
print(summary.to_string(index=False))

out = ROOT / "reports" / "cv5_variant_summary.csv"
summary.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")

runs_out = ROOT / "reports" / "cv5_variant_runs.csv"
cv.to_csv(runs_out, index=False, encoding="utf-8-sig")
print(f"[OK] wrote -> {runs_out}")