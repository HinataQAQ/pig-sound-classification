from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "mctafd_ablation_summary.csv"

df = pd.read_csv(CSV)

ab = df[df["experiment"].str.startswith("cv5_ablation_", na=False)].copy()

def parse_fold(exp):
    m = re.search(r"cv5_ablation_fold(\d+)_", exp)
    return int(m.group(1)) if m else None

def parse_model(exp):
    s = exp
    # Order matters.
    if "_mctafd_no_se_" in s:
        return "mctafd_no_se"
    if "_mel_mfcc_dyn_flux_" in s:
        return "mel_mfcc_dyn_flux"
    if "_mel_mfcc_dyn_" in s:
        return "mel_mfcc_dyn"
    if "_mel_mfcc_" in s:
        return "mel_mfcc"
    if "_mctafd_" in s:
        return "mctafd_se"
    if "_logmel_" in s:
        return "logmel"
    return "unknown"

ab["fold"] = ab["experiment"].apply(parse_fold)
ab["model"] = ab["experiment"].apply(parse_model)

cols = [
    "experiment", "model", "fold", "seed",
    "n_mels", "fmax", "best_val_macro_f1", "test_macro_f1"
]

ab = ab[cols].sort_values(["model", "seed", "fold"])

print("\n[CV5 ABLATION RUNS]")
print(ab.to_string(index=False))

summary = (
    ab.groupby("model")
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

print("\n[CV5 ABLATION SUMMARY]")
print(summary.to_string(index=False))

out = ROOT / "reports" / "cv5_ablation_summary.csv"
summary.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")