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

# 原始 dedup CV5 Log-Mel
for s in REPORTS.glob("cv5_fold*_logmel_nmel64_fmax8000_seed*/summary.json"):
    exp = s.parent.name
    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "model": "cv5_logmel_baseline",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
    })

# cap3x Log-Mel
for s in REPORTS.glob("cv5_expanded_cap3x_fold*_logmel_seed*/summary.json"):
    exp = s.parent.name
    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "model": "cv5_cap3x_logmel",
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

if "cv5_logmel_baseline" not in wide.columns or "cv5_cap3x_logmel" not in wide.columns:
    raise RuntimeError(f"Missing columns. Current columns={list(wide.columns)}")

delta = wide["cv5_cap3x_logmel"].values - wide["cv5_logmel_baseline"].values

rng = np.random.default_rng(3407)
boots = []

for _ in range(10000):
    sample = rng.choice(delta, size=len(delta), replace=True)
    boots.append(sample.mean())

ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

try:
    stat, p = wilcoxon(
        wide["cv5_cap3x_logmel"],
        wide["cv5_logmel_baseline"],
    )
except ValueError:
    p = np.nan

summary = {
    "comparison": "cv5_cap3x_logmel_minus_cv5_baseline_logmel",
    "n": len(wide),
    "baseline_mean": float(wide["cv5_logmel_baseline"].mean()),
    "cap3x_mean": float(wide["cv5_cap3x_logmel"].mean()),
    "mean_delta": float(delta.mean()),
    "std_delta": float(delta.std(ddof=1)),
    "ci95_low": float(ci_low),
    "ci95_high": float(ci_high),
    "wilcoxon_p": float(p),
}

out = ROOT / "reports" / "cv5_cap3x_logmel_vs_baseline_paired_stats.csv"
pd.DataFrame([summary]).to_csv(out, index=False, encoding="utf-8-sig")

print("\n[PAIRED: cap3x Log-Mel - baseline CV5 Log-Mel]")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n[PAIRED RUNS]")
print(wide.to_string())

print(f"\n[OK] wrote -> {out}")