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


def collect_runs():
    rows = []

    # Baseline CV10 Log-Mel:
    # reports/cv10_fold{fold}_logmel_seed{seed}/summary.json
    for s in REPORTS.glob("cv10_fold*_logmel_seed*/summary.json"):
        exp = s.parent.name
        with open(s, "r", encoding="utf-8") as f:
            m = json.load(f)

        rows.append({
            "experiment": exp,
            "model": "logmel_baseline",
            "fold": parse_fold(exp),
            "seed": parse_seed(exp),
            "test_macro_f1": m.get("test_macro_f1"),
            "best_val_macro_f1": m.get("best_val_macro_f1"),
        })

    # Pretrain CV10 Log-Mel:
    # reports/cv10_pretrain_no_feeding_fold{fold}_logmel_seed{seed}/summary.json
    for s in REPORTS.glob("cv10_pretrain_no_feeding_fold*_logmel_seed*/summary.json"):
        exp = s.parent.name
        with open(s, "r", encoding="utf-8") as f:
            m = json.load(f)

        rows.append({
            "experiment": exp,
            "model": "logmel_pretrain",
            "fold": parse_fold(exp),
            "seed": parse_seed(exp),
            "test_macro_f1": m.get("test_macro_f1"),
            "best_val_macro_f1": m.get("best_val_macro_f1"),
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No CV10 Log-Mel baseline/pretrain summary.json files found.")

    for c in ["fold", "seed", "test_macro_f1", "best_val_macro_f1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["test_macro_f1"].notna()].copy()

    return df


def main():
    df = collect_runs()

    runs_out = ROOT / "reports" / "cv10_logmel_pretrain_vs_baseline_runs.csv"
    df.to_csv(runs_out, index=False, encoding="utf-8-sig")

    wide = df.pivot_table(
        index=["fold", "seed"],
        columns="model",
        values="test_macro_f1",
        aggfunc="first",
    ).dropna()

    if "logmel_baseline" not in wide.columns or "logmel_pretrain" not in wide.columns:
        raise RuntimeError(
            f"Need both logmel_baseline and logmel_pretrain. Current columns={list(wide.columns)}"
        )

    delta = wide["logmel_pretrain"].values - wide["logmel_baseline"].values

    rng = np.random.default_rng(3407)
    boots = []

    for _ in range(10000):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boots.append(sample.mean())

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    try:
        stat, p = wilcoxon(wide["logmel_pretrain"], wide["logmel_baseline"])
    except ValueError:
        p = np.nan

    summary = {
        "comparison": "logmel_pretrain_minus_logmel_baseline",
        "n": len(wide),
        "baseline_mean": float(wide["logmel_baseline"].mean()),
        "pretrain_mean": float(wide["logmel_pretrain"].mean()),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_p": float(p),
    }

    out = ROOT / "reports" / "cv10_logmel_pretrain_vs_baseline_paired_stats.csv"
    pd.DataFrame([summary]).to_csv(out, index=False, encoding="utf-8-sig")

    print("\n[PAIRED STATS: Log-Mel-pretrain - Log-Mel-baseline]")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n[PAIRED RUNS]")
    print(wide.to_string())

    print(f"\n[OK] wrote -> {runs_out}")
    print(f"[OK] wrote -> {out}")


if __name__ == "__main__":
    main()