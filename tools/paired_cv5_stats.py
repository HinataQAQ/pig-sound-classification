from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "mctafd_ablation_summary.csv"

df = pd.read_csv(CSV)

def parse_fold(exp):
    m = re.search(r"cv5_fold(\d+)_", exp)
    return int(m.group(1)) if m else None

def parse_model(exp):
    if "_logmel_" in exp:
        return "logmel"
    if "_mctafd_no_se_" in exp:
        return "mctafd_no_se"
    if "_mctafd_se_" in exp:
        return "mctafd_se"
    return None

cv = df[df["experiment"].str.startswith("cv5_fold", na=False)].copy()
cv["fold"] = cv["experiment"].apply(parse_fold)
cv["model"] = cv["experiment"].apply(parse_model)
cv = cv[cv["model"].isin(["logmel", "mctafd_no_se"])]

wide = cv.pivot_table(
    index=["fold", "seed"],
    columns="model",
    values="test_macro_f1",
    aggfunc="first",
).dropna()

wide["delta_mctafd_no_se_minus_logmel"] = wide["mctafd_no_se"] - wide["logmel"]

d = wide["delta_mctafd_no_se_minus_logmel"].values

rng = np.random.default_rng(3407)
boots = []
for _ in range(10000):
    sample = rng.choice(d, size=len(d), replace=True)
    boots.append(sample.mean())

ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

try:
    stat, p = wilcoxon(wide["mctafd_no_se"], wide["logmel"])
except ValueError:
    stat, p = np.nan, np.nan

print("[PAIRED CV5]")
print(wide.to_string())

print("\n[DELTA: MCTAFD-noSE - Log-Mel]")
print(f"n          = {len(d)}")
print(f"mean_delta = {d.mean():.6f}")
print(f"std_delta  = {d.std(ddof=1):.6f}")
print(f"95% CI     = [{ci_low:.6f}, {ci_high:.6f}]")
print(f"wilcoxon_p = {p:.6f}")

out = ROOT / "reports" / "cv5_paired_logmel_vs_mctafd_no_se.csv"
wide.to_csv(out, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")