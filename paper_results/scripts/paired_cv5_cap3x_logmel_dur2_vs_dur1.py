from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_fold(exp: str):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_seed(exp: str):
    m = re.search(r"seed(\d+)", exp)
    return int(m.group(1)) if m else None


rows = []

# cap3x Log-Mel dur1 baseline
for s in REPORTS.glob("cv5_expanded_cap3x_fold*_logmel_seed*/summary.json"):
    exp = s.parent.name

    # 排除 logmel_dur2 / logmel_dur3 / logmel_specaug / etc.
    if "_logmel_seed" not in exp:
        continue

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "model": "logmel_dur1",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
    })


# cap3x Log-Mel dur2
for s in REPORTS.glob("cv5_expanded_cap3x_fold*_logmel_dur2_seed*/summary.json"):
    exp = s.parent.name

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "model": "logmel_dur2",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
    })


df = pd.DataFrame(rows)

for c in ["fold", "seed", "test_macro_f1"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

wide = df.pivot_table(
    index=["fold", "seed"],
    columns="model",
    values="test_macro_f1",
    aggfunc="first",
).dropna()

if "logmel_dur1" not in wide.columns or "logmel_dur2" not in wide.columns:
    raise RuntimeError(f"Missing required columns. Current columns={list(wide.columns)}")

delta = wide["logmel_dur2"].values - wide["logmel_dur1"].values

rng = np.random.default_rng(3407)
boots = []

for _ in range(10000):
    sample = rng.choice(delta, size=len(delta), replace=True)
    boots.append(sample.mean())

ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

try:
    stat, p = wilcoxon(
        wide["logmel_dur2"],
        wide["logmel_dur1"],
    )
except ValueError:
    p = np.nan

summary = {
    "comparison": "cap3x_logmel_dur2_minus_dur1",
    "n": len(wide),
    "dur1_mean": float(wide["logmel_dur1"].mean()),
    "dur2_mean": float(wide["logmel_dur2"].mean()),
    "mean_delta": float(delta.mean()),
    "std_delta": float(delta.std(ddof=1)),
    "ci95_low": float(ci_low),
    "ci95_high": float(ci_high),
    "wilcoxon_p": float(p),
}

out = ROOT / "reports" / "cv5_cap3x_logmel_dur2_vs_dur1_paired_stats.csv"
pd.DataFrame([summary]).to_csv(out, index=False, encoding="utf-8-sig")

print("\n[PAIRED: cap3x Log-Mel dur2 - dur1]")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n[PAIRED RUNS]")
print(wide.to_string())

print(f"\n[OK] wrote -> {out}")