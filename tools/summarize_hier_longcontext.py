from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_fold(exp):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_seed(exp):
    m = re.search(r"seed(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_model(exp):
    if "_logmel_dur2_hier_w05_" in exp:
        return "logmel_dur2_hier_w05"
    if "_logmel_dur2_hier_w02_" in exp:
        return "logmel_dur2_hier_w02"
    if "_logmel_dur2_hier_w10_" in exp:
        return "logmel_dur2_hier_w10"
    return None


rows = []

for s in REPORTS.glob("cv5_expanded_cap3x_fold*_logmel_dur2_hier_w*_seed*/summary.json"):
    exp = s.parent.name
    model = parse_model(exp)

    if model is None:
        continue

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "experiment": exp,
        "model": model,
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_macro_f1": m.get("test_macro_f1"),
        "test_acc": m.get("test_acc"),
        "test_aux_macro_f1": m.get("test_aux_macro_f1"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "best_epoch": m.get("best_epoch"),
        "hier_aux_weight": m.get("hier_aux_weight"),
    })


df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No hierarchical long-context results found.")

for c in [
    "fold",
    "seed",
    "test_macro_f1",
    "test_acc",
    "test_aux_macro_f1",
    "best_val_macro_f1",
    "best_epoch",
    "hier_aux_weight",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values(["model", "seed", "fold"])

runs_out = REPORTS / "hier_longcontext_runs.csv"
df.to_csv(runs_out, index=False, encoding="utf-8-sig")

summary = (
    df.groupby("model")
      .agg(
          n=("test_macro_f1", "count"),
          mean_macro_f1=("test_macro_f1", "mean"),
          std_macro_f1=("test_macro_f1", "std"),
          min_macro_f1=("test_macro_f1", "min"),
          max_macro_f1=("test_macro_f1", "max"),
          mean_aux_f1=("test_aux_macro_f1", "mean"),
          mean_best_val=("best_val_macro_f1", "mean"),
          mean_best_epoch=("best_epoch", "mean"),
      )
      .reset_index()
      .sort_values("mean_macro_f1", ascending=False)
)

summary_out = REPORTS / "hier_longcontext_summary.csv"
summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

print("\n[HIER LONG-CONTEXT RUNS]")
print(df.to_string(index=False))

print("\n[HIER LONG-CONTEXT SUMMARY]")
print(summary.to_string(index=False))

print(f"\n[OK] wrote -> {runs_out}")
print(f"[OK] wrote -> {summary_out}")