from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]

def parse_fold(exp: str):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None

def parse_model(exp: str):
    if exp.startswith("cv5_gated_") and "_mctafd_no_se_" in exp and "boundary" not in exp:
        return "mctafd_no_se_gated"

    if exp.startswith("cv5_fold") and "_logmel_" in exp:
        return "logmel"

    if exp.startswith("cv5_fold") and "_mctafd_no_se_" in exp:
        return "mctafd_no_se_direct"

    if exp.startswith("cv5_fold") and "_mctafd_se_" in exp:
        return "mctafd_se"

    return None

rows = []

for s in REPORTS.glob("cv5*/summary.json"):
    exp = s.parent.name
    model = parse_model(exp)
    if model is None:
        continue

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    cm = m.get("confusion")
    if cm is None:
        continue

    rows.append({
        "experiment": exp,
        "model": model,
        "fold": parse_fold(exp),
        "seed": m.get("seed"),
        "test_macro_f1": m.get("test_macro_f1"),
        "confusion": np.array(cm, dtype=np.int64),
    })

df = pd.DataFrame(rows)

summary_rows = []

print("\n[AGGREGATED CONFUSIONS]")

for model, g in df.groupby("model"):
    cm_sum = np.stack(g["confusion"].values, axis=0).sum(axis=0)

    print(f"\nMODEL = {model}")
    print("n =", len(g))
    print(pd.DataFrame(cm_sum, index=LABELS, columns=LABELS).to_string())

    total = cm_sum.sum()
    acc = np.trace(cm_sum) / total

    recalls = {}
    precisions = {}
    f1s = {}

    for i, lab in enumerate(LABELS):
        tp = cm_sum[i, i]
        fn = cm_sum[i, :].sum() - tp
        fp = cm_sum[:, i].sum() - tp

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

        recalls[lab] = rec
        precisions[lab] = pre
        f1s[lab] = f1

    macro_f1_from_cm = float(np.mean(list(f1s.values())))

    feeding_idx = LABELS.index("feeding")
    stress_idx = LABELS.index("stress_vocal")

    feeding_to_stress = int(cm_sum[feeding_idx, stress_idx])
    stress_to_feeding = int(cm_sum[stress_idx, feeding_idx])

    row = {
        "model": model,
        "n_runs": len(g),
        "acc_from_cm": acc,
        "macro_f1_from_cm": macro_f1_from_cm,
        "feeding_recall": recalls["feeding"],
        "feeding_f1": f1s["feeding"],
        "stress_recall": recalls["stress_vocal"],
        "stress_f1": f1s["stress_vocal"],
        "feeding_to_stress": feeding_to_stress,
        "stress_to_feeding": stress_to_feeding,
    }

    summary_rows.append(row)

    print("\nmetrics:")
    for k, v in row.items():
        print(f"  {k}: {v}")

out = ROOT / "reports" / "cv5_variant_confusion_summary.csv"
pd.DataFrame(summary_rows).to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")