from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "reports" / "cv5_variant_runs.csv"

df = pd.read_csv(RUNS_CSV)

required = {
    "logmel",
    "mctafd_no_se_direct",
    "mctafd_se",
    "mctafd_no_se_gated",
}

df = df[df["model"].isin(required)].copy()

wide = df.pivot_table(
    index=["fold", "seed"],
    columns="model",
    values="test_macro_f1",
    aggfunc="first",
).dropna()

pairs = [
    ("mctafd_no_se_gated", "logmel"),
    ("mctafd_no_se_gated", "mctafd_no_se_direct"),
    ("mctafd_no_se_gated", "mctafd_se"),
    ("mctafd_no_se_direct", "logmel"),
    ("mctafd_se", "logmel"),
]

rng = np.random.default_rng(3407)

rows = []

print("\n[PAIRED CV5 VARIANT STATS]")
print(f"paired units = {len(wide)}")
print(wide.to_string())

for a, b in pairs:
    if a not in wide.columns or b not in wide.columns:
        continue

    delta = wide[a].values - wide[b].values

    boots = []
    for _ in range(10000):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boots.append(sample.mean())

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    try:
        stat, p = wilcoxon(wide[a], wide[b])
    except ValueError:
        stat, p = np.nan, np.nan

    row = {
        "model_a": a,
        "model_b": b,
        "n": len(delta),
        "mean_a": float(wide[a].mean()),
        "mean_b": float(wide[b].mean()),
        "mean_delta_a_minus_b": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_p": float(p),
    }

    rows.append(row)

    print(f"\n[{a} - {b}]")
    print(f"mean_a     = {row['mean_a']:.6f}")
    print(f"mean_b     = {row['mean_b']:.6f}")
    print(f"mean_delta = {row['mean_delta_a_minus_b']:.6f}")
    print(f"std_delta  = {row['std_delta']:.6f}")
    print(f"95% CI     = [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]")
    print(f"wilcoxon_p = {row['wilcoxon_p']:.6f}")

out = ROOT / "reports" / "cv5_variant_paired_stats.csv"
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")