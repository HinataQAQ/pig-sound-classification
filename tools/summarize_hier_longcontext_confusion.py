from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]


def collect(pattern, model_name):
    cms = []

    for s in REPORTS.glob(pattern):
        exp = s.parent.name

        if "_attn_" in exp:
            continue

        with open(s, "r", encoding="utf-8") as f:
            m = json.load(f)

        if "confusion" not in m:
            continue

        cms.append(np.array(m["confusion"], dtype=np.int64))

    return model_name, cms


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
            "logmel_dur2",
            "cv5_expanded_cap3x_fold*_logmel_dur2_seed*/summary.json",
        ),
        (
            "logmel_dur2_hier_w05",
            "cv5_expanded_cap3x_fold*_logmel_dur2_hier_w05_seed*/summary.json",
        ),
    ]

    rows = []

    for model_name, pattern in configs:
        name, cms = collect(pattern, model_name)

        if len(cms) == 0:
            print(f"[WARN] no confusion found for {name}")
            continue

        cm_sum = np.stack(cms, axis=0).sum(axis=0)

        print(f"\n[CONFUSION] {name}")
        print(pd.DataFrame(cm_sum, index=LABELS, columns=LABELS).to_string())

        row = {
            "model": name,
            "n_runs": len(cms),
        }

        row.update(cm_metrics(cm_sum))
        rows.append(row)

    out = REPORTS / "hier_longcontext_confusion_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n[OK] wrote -> {out}")


if __name__ == "__main__":
    main()