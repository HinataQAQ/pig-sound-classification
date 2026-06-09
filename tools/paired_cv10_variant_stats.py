from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "reports" / "cv10_variant_runs.csv"

df = pd.read_csv(RUNS_CSV)

df = df[df["model"].isin(["logmel", "mctafd_no_se_gated"])].copy()

wide = df.pivot_table(
    index=["fold", "seed"],
    columns="model",
    values="test_macro_f1",
    aggfunc="first",
).dropna()

delta = wide["mctafd_no_se_gated"].values - wide["logmel"].values

rng = np.random.default_rng(3407)
boots = []
for _ in range(10000):
    sample = rng.choice(delta, size=len(delta), replace=True)
    boots.append(sample.mean())

ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

try:
    stat, p = wilcoxon(wide["mctafd_no_se_gated"], wide["logmel"])
except ValueError:
    stat, p = np.nan, np.nan

print("\n[CV10 PAIRED STATS]")
print(wide.to_string())

print("\n[mctafd_no_se_gated - logmel]")
print(f"n          = {len(delta)}")
print(f"mean_delta = {delta.mean():.6f}")
print(f"std_delta  = {delta.std(ddof=1):.6f}")
print(f"95% CI     = [{ci_low:.6f}, {ci_high:.6f}]")
print(f"wilcoxon_p = {p:.6f}")

out = ROOT / "reports" / "cv10_paired_gated_vs_logmel.csv"
wide.to_csv(out, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")