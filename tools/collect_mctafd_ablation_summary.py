from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
REPORT_ROOT = ROOT / "reports"

rows = []

for d in REPORT_ROOT.glob("ablation_*"):
    if not d.is_dir():
        continue

    s = d / "summary.json"
    if not s.exists():
        continue

    with open(s, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "experiment": d.name,
        "feature_mode": m.get("feature_mode"),
        "use_se": m.get("use_se"),
        "seed": m.get("seed"),
        "best_epoch": m.get("best_epoch"),
        "best_val_macro_f1": m.get("best_val_macro_f1"),
        "test_acc": m.get("test_acc"),
        "test_macro_f1": m.get("test_macro_f1"),
    })

df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No ablation summary.json found under reports/ablation_*")

df = df.sort_values(["experiment"])

out = REPORT_ROOT / "mctafd_ablation_summary.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print(df.to_string(index=False))
print("[OK] wrote ->", out)