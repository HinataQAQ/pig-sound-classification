from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]


def parse_model(exp):
    s = exp.lower()
    if s.startswith("cv10_") and "_logmel_" in s:
        return "logmel"
    if s.startswith("cv10_") and "_mctafd_no_se_gated_" in s:
        return "mctafd_no_se_gated"
    return None


rows = []

for summary_path in REPORTS.glob("cv10*/summary.json"):
    exp = summary_path.parent.name
    model = parse_model(exp)
    if model is None:
        continue

    with open(summary_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    cm = m.get("confusion")
    if cm is None:
        continue

    rows.append({
        "experiment": exp,
        "model": model,
        "confusion": np.array(cm, dtype=np.int64),
    })

df = pd.DataFrame(rows)

summary_rows = []

for model, g in df.groupby("model"):
    cm_sum = np.stack(g["confusion"].values, axis=0).sum(axis=0)

    print(f"\nMODEL = {model}")
    print(pd.DataFrame(cm_sum, index=LABELS, columns=LABELS).to_string())

    f1s = {}
    recalls = {}

    for i, lab in enumerate(LABELS):
        tp = cm_sum[i, i]
        fn = cm_sum[i, :].sum() - tp
        fp = cm_sum[:, i].sum() - tp

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        recalls[lab] = recall
        f1s[lab] = f1

    feeding_idx = LABELS.index("feeding")
    stress_idx = LABELS.index("stress_vocal")

    summary_rows.append({
        "model": model,
        "n_runs": len(g),
        "macro_f1_from_cm": float(np.mean(list(f1s.values()))),
        "feeding_f1": f1s["feeding"],
        "stress_f1": f1s["stress_vocal"],
        "feeding_recall": recalls["feeding"],
        "stress_recall": recalls["stress_vocal"],
        "feeding_to_stress": int(cm_sum[feeding_idx, stress_idx]),
        "stress_to_feeding": int(cm_sum[stress_idx, feeding_idx]),
    })

out = ROOT / "reports" / "cv10_confusion_summary.csv"
pd.DataFrame(summary_rows).to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")