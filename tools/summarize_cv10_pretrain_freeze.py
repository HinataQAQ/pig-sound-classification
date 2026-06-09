from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_fold(exp: str):
    m = re.search(r"fold(\d+)", exp)
    return int(m.group(1)) if m else None


def parse_seed(exp: str):
    m = re.search(r"seed(\d+)", exp)
    return int(m.group(1)) if m else None


rows = []

for s in REPORTS.glob("cv10_pretrain_freeze5_lr1e4_fold*_logmel_seed*/summary.json"):
    exp = s.parent.name

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "experiment": exp,
        "model": "logmel_pretrain_freeze5_lr1e4",
        "fold": parse_fold(exp),
        "seed": parse_seed(exp),
        "test_acc": m.get("test_acc"),
        "test_macro_f1": m.get("test_macro_f1"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "best_epoch": m.get("best_epoch"),
        "pretrained_ckpt": m.get("pretrained_ckpt"),
        "freeze_encoder_epochs": m.get("freeze_encoder_epochs"),
    })

df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No cv10_pretrain_freeze5_lr1e4 results found.")

for c in ["fold", "seed", "test_acc", "test_macro_f1", "best_val_macro_f1", "best_epoch", "freeze_encoder_epochs"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values(["seed", "fold"])

runs_out = REPORTS / "cv10_pretrain_freeze5_lr1e4_runs.csv"
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
          mean_best_epoch=("best_epoch", "mean"),
      )
      .reset_index()
)

summary_out = REPORTS / "cv10_pretrain_freeze5_lr1e4_summary.csv"
summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

print("\n[FREEZE5 LR1E-4 RUNS]")
print(df.to_string(index=False))

print("\n[FREEZE5 LR1E-4 SUMMARY]")
print(summary.to_string(index=False))

print(f"\n[OK] wrote -> {runs_out}")
print(f"[OK] wrote -> {summary_out}")