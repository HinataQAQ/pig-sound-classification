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


def read_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


rows = []

# Baseline: cap3x Log-Mel dur2
for s in REPORTS.glob("cv5_expanded_cap3x_fold*_logmel_dur2_seed*/summary.json"):
    exp = s.parent.name

    # 排除 attention / hier / 其它变体
    if "_attn_" in exp or "_hier_" in exp:
        continue

    m = read_summary(s)

    rows.append({
        "experiment": exp,
        "model": "logmel_dur2",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
    })


# Hierarchical: cap3x Log-Mel dur2 + subtype auxiliary head, w=0.5
for s in REPORTS.glob("cv5_expanded_cap3x_fold*_logmel_dur2_hier_w05_seed*/summary.json"):
    exp = s.parent.name
    m = read_summary(s)

    rows.append({
        "experiment": exp,
        "model": "logmel_dur2_hier_w05",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
    })


df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No matching dur2 / hier_w05 summary.json files found.")

for c in ["fold", "seed", "test_macro_f1"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["test_macro_f1"].notna()].copy()

wide = df.pivot_table(
    index=["fold", "seed"],
    columns="model",
    values="test_macro_f1",
    aggfunc="first",
)

print("\n[PAIR COVERAGE]")
print(wide.to_string())

missing = wide[wide.isna().any(axis=1)]

if len(missing) > 0:
    print("\n[MISSING PAIRS]")
    print(missing.to_string())

wide = wide.dropna()

if "logmel_dur2" not in wide.columns or "logmel_dur2_hier_w05" not in wide.columns:
    raise RuntimeError(f"Missing required columns. Current columns={list(wide.columns)}")

delta = wide["logmel_dur2_hier_w05"].values - wide["logmel_dur2"].values

rng = np.random.default_rng(3407)
boots = []

for _ in range(10000):
    sample = rng.choice(delta, size=len(delta), replace=True)
    boots.append(sample.mean())

ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

try:
    stat, p = wilcoxon(
        wide["logmel_dur2_hier_w05"],
        wide["logmel_dur2"],
    )
except ValueError:
    p = np.nan

summary = {
    "comparison": "logmel_dur2_hier_w05_minus_logmel_dur2",
    "n": int(len(wide)),
    "dur2_mean": float(wide["logmel_dur2"].mean()),
    "hier_w05_mean": float(wide["logmel_dur2_hier_w05"].mean()),
    "mean_delta": float(delta.mean()),
    "std_delta": float(delta.std(ddof=1)),
    "ci95_low": float(ci_low),
    "ci95_high": float(ci_high),
    "wilcoxon_p": float(p),
}

out = ROOT / "reports" / "cv5_cap3x_hier_w05_vs_dur2_paired_stats.csv"
pd.DataFrame([summary]).to_csv(out, index=False, encoding="utf-8-sig")

runs_out = ROOT / "reports" / "cv5_cap3x_hier_w05_vs_dur2_paired_runs.csv"
wide.to_csv(runs_out, encoding="utf-8-sig")

print("\n[PAIRED: hier_w05 - dur2]")
for k, v in summary.items():
    print(f"{k}: {v}")

print(f"\n[OK] wrote -> {out}")
print(f"[OK] wrote -> {runs_out}")