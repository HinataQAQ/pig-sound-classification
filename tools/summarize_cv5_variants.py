from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

def parse_fold(exp: str):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None

def parse_weight(exp: str):
    if "_w02_" in exp:
        return "w02"
    if "_w05_" in exp:
        return "w05"
    if "_w10_" in exp:
        return "w10"
    return None

def parse_model(exp: str):
    # Order matters: more specific names first.
    if exp.startswith("cv5_gated_boundary_") and "_mctafd_no_se_" in exp:
        w = parse_weight(exp)
        return f"mctafd_no_se_gated_boundary_{w}" if w else "mctafd_no_se_gated_boundary"

    if exp.startswith("cv5_gated_") and "_mctafd_no_se_" in exp:
        return "mctafd_no_se_gated"

    if exp.startswith("cv5_boundary_") and "_logmel_" in exp:
        w = parse_weight(exp)
        return f"logmel_boundary_{w}" if w else "logmel_boundary"

    if exp.startswith("cv5_boundary_") and "_mctafd_no_se_" in exp:
        w = parse_weight(exp)
        return f"mctafd_no_se_boundary_{w}" if w else "mctafd_no_se_boundary"

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

    gate = m.get("aux_gate")
    gate_mean = None
    gate_max = None

    if isinstance(gate, list) and len(gate) > 0:
        gate_mean = sum(float(x) for x in gate) / len(gate)
        gate_max = max(float(x) for x in gate)

    rows.append({
        "experiment": exp,
        "model": model,
        "fold": parse_fold(exp),
        "seed": m.get("seed"),
        "feature_mode": m.get("feature_mode"),
        "use_se": m.get("use_se"),
        "n_mels": m.get("n_mels"),
        "fmax": m.get("fmax"),
        "boundary_loss_weight": m.get("boundary_loss_weight"),
        "boundary_warmup_epochs": m.get("boundary_warmup_epochs"),
        "fusion_type": m.get("fusion_type"),
        "aux_gate_init": m.get("aux_gate_init"),
        "gate_mean": gate_mean,
        "gate_max": gate_max,
        "best_epoch": m.get("best_epoch"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "test_acc": m.get("test_acc"),
        "test_macro_f1": m.get("test_macro_f1"),
    })

if len(rows) == 0:
    raise RuntimeError("No recognized cv5 variant summary.json files found.")

df = pd.DataFrame(rows)

df = df.sort_values(["model", "seed", "fold", "experiment"])

runs_out = REPORTS / "cv5_variant_runs.csv"
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
          mean_gate=("gate_mean", "mean"),
          max_gate=("gate_max", "max"),
      )
      .reset_index()
      .sort_values("mean_macro_f1", ascending=False)
)

summary_out = REPORTS / "cv5_variant_summary.csv"
summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

print("\n[CV5 VARIANT SUMMARY]")
print(summary.to_string(index=False))

print(f"\n[OK] wrote -> {runs_out}")
print(f"[OK] wrote -> {summary_out}")
