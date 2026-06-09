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

for s in REPORTS.glob("cv10_fold*_logmel_seed*/summary.json"):
    exp = s.parent.name
    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "model": "baseline_logmel",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
    })

for s in REPORTS.glob("cv10_pretrain_freeze5_lr1e4_fold*_logmel_seed*/summary.json"):
    exp = s.parent.name
    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "model": "logmel_pretrain_freeze5_lr1e4",
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

if "baseline_logmel" not in wide.columns or "logmel_pretrain_freeze5_lr1e4" not in wide.columns:
    raise RuntimeError(f"Missing required columns. Current columns={list(wide.columns)}")

delta = wide["logmel_pretrain_freeze5_lr1e4"].values - wide["baseline_logmel"].values

rng = np.random.default_rng(3407)
boots = []

for _ in range(10000):
    sample = rng.choice(delta, size=len(delta), replace=True)
    boots.append(sample.mean())

ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

try:
    stat, p = wilcoxon(
        wide["logmel_pretrain_freeze5_lr1e4"],
        wide["baseline_logmel"],
    )
except ValueError:
    p = np.nan

summary = {
    "comparison": "freeze5_lr1e4_pretrain_minus_baseline",
    "n": len(wide),
    "baseline_mean": float(wide["baseline_logmel"].mean()),
    "pretrain_freeze_mean": float(wide["logmel_pretrain_freeze5_lr1e4"].mean()),
    "mean_delta": float(delta.mean()),
    "std_delta": float(delta.std(ddof=1)),
    "ci95_low": float(ci_low),
    "ci95_high": float(ci_high),
    "wilcoxon_p": float(p),
}

out = ROOT / "reports" / "cv10_pretrain_freeze5_lr1e4_vs_baseline_paired_stats.csv"
pd.DataFrame([summary]).to_csv(out, index=False, encoding="utf-8-sig")

print("\n[PAIRED: freeze5 lr1e-4 pretrain - baseline]")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n[PAIRED RUNS]")
print(wide.to_string())

print(f"\n[OK] wrote -> {out}")