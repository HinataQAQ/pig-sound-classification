from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]


def parse_fold(exp):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_seed(exp):
    m = re.search(r"seed(\d+)", exp)
    return int(m.group(1)) if m else None


def collect(pattern, model_name):
    rows = []
    cms = []

    for s in REPORTS.glob(pattern):
        exp = s.parent.name

        with open(s, "r", encoding="utf-8") as f:
            m = json.load(f)

        cm = np.array(m["confusion"], dtype=np.int64)

        rows.append({
            "experiment": exp,
            "model": model_name,
            "fold": parse_fold(exp),
            "seed": parse_seed(exp),
            "test_macro_f1": m.get("test_macro_f1"),
        })

        cms.append(cm)

    return rows, cms


def cm_metrics(cm):
    f1s = {}
    recalls = {}

    for i, lab in enumerate(LABELS):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recalls[lab] = recall
        f1s[lab] = f1

    feeding_idx = LABELS.index("feeding")
    stress_idx = LABELS.index("stress_vocal")

    return {
        "macro_f1_from_cm": float(np.mean(list(f1s.values()))),
        "feeding_f1": float(f1s["feeding"]),
        "stress_f1": float(f1s["stress_vocal"]),
        "feeding_recall": float(recalls["feeding"]),
        "stress_recall": float(recalls["stress_vocal"]),
        "feeding_to_stress": int(cm[feeding_idx, stress_idx]),
        "stress_to_feeding": int(cm[stress_idx, feeding_idx]),
    }


def main():
    configs = [
        (
            "baseline_cv5_logmel",
            "cv5_fold*_logmel_nmel64_fmax8000_seed*/summary.json",
        ),
        (
            "cap3x_logmel",
            "cv5_expanded_cap3x_fold*_logmel_seed*/summary.json",
        ),
    ]

    summary_rows = []

    for model_name, pattern in configs:
        rows, cms = collect(pattern, model_name)

        if len(cms) == 0:
            print(f"[WARN] no results for {model_name}")
            continue

        cm_sum = np.stack(cms, axis=0).sum(axis=0)

        print(f"\n[CONFUSION] {model_name}")
        print(pd.DataFrame(cm_sum, index=LABELS, columns=LABELS).to_string())

        row = {
            "model": model_name,
            "n_runs": len(cms),
        }
        row.update(cm_metrics(cm_sum))
        summary_rows.append(row)

    out = REPORTS / "cv5_cap3x_logmel_confusion_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n[OK] wrote -> {out}")


if __name__ == "__main__":
    main()