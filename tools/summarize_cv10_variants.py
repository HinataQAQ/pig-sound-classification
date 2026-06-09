from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_fold(exp):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_model(exp):
    s = exp.lower()

    if "_mctafd_no_se_gated_" in s:
        return "mctafd_no_se_gated"

    if "_logmel_" in s:
        return "logmel"

    return None


rows = []

for summary_path in REPORTS.glob("cv10*/summary.json"):
    exp = summary_path.parent.name
    model = parse_model(exp)

    if model is None:
        continue

    with open(summary_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    gate = m.get("aux_gate")
    gate_mean = None
    gate_max = None

    if isinstance(gate, list) and len(gate) > 0:
        vals = [float(x) for x in gate]
        gate_mean = sum(vals) / len(vals)
        gate_max = max(vals)

    rows.append({
        "experiment": exp,
        "model": model,
        "fold": parse_fold(exp),
        "seed": m.get("seed"),
        "test_acc": m.get("test_acc"),
        "test_macro_f1": m.get("test_macro_f1"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "fusion_type": m.get("fusion_type"),
        "aux_gate_init": m.get("aux_gate_init"),
        "gate_mean": gate_mean,
        "gate_max": gate_max,
    })


df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No CV10 summary.json found.")

for c in ["fold", "seed", "test_acc", "test_macro_f1", "best_val_macro_f1", "gate_mean", "gate_max"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["test_macro_f1"].notna()].copy()
df = df.sort_values(["model", "seed", "fold", "experiment"])

runs_out = REPORTS / "cv10_variant_runs.csv"
df.to_csv(runs_out, index=False, encoding="utf-8-sig")

summary = (
    df.groupby("model")
      .agg(
          n=("test_macro_f1", "count"),
          mean_macro_f1=("test_macro_f1", "mean"),
          std_macro_f1=("test_macro_f1", "std"),
          min_macro_f1=("test_macro_f1", "min"),
          max_macro_f1=("test_macro_f1", "max"),
          mean_best_val=("best_val_macro_f1", "mean"),
          mean_test_acc=("test_acc", "mean"),
          mean_gate=("gate_mean", "mean"),
          max_gate=("gate_max", "max"),
      )
      .reset_index()
      .sort_values("mean_macro_f1", ascending=False)
)

summary_out = REPORTS / "cv10_variant_summary.csv"
summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

print("\n[CV10 RUNS]")
print(df.to_string(index=False))

print("\n[CV10 SUMMARY]")
print(summary.to_string(index=False))

print(f"\n[OK] wrote -> {runs_out}")
print(f"[OK] wrote -> {summary_out}")