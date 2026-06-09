from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

rows = []

for d in REPORTS.glob("cv5_gated*"):
    s = d / "summary.json"
    if not s.exists():
        continue

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    gate = m.get("aux_gate")
    if gate is None:
        continue

    row = {
        "experiment": d.name,
        "seed": m.get("seed"),
        "test_macro_f1": m.get("test_macro_f1"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
    }

    for i, v in enumerate(gate):
        row[f"gate_{i+1}"] = v

    row["gate_mean"] = float(np.mean(gate))
    row["gate_max"] = float(np.max(gate))
    rows.append(row)

df = pd.DataFrame(rows)
if len(df) == 0:
    raise RuntimeError("No aux_gate found.")

print(df.to_string(index=False))

summary = df[[c for c in df.columns if c.startswith("gate_")]].describe()
print("\n[GATE SUMMARY]")
print(summary.to_string())

out = REPORTS / "cv5_gated_aux_gate_summary.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[OK] wrote -> {out}")