from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
OUT_ROOT = ROOT / "reports" / "feature_ablation"

rows = []

if not OUT_ROOT.exists():
    raise RuntimeError(f"Missing folder: {OUT_ROOT}")

for feature_dir in OUT_ROOT.iterdir():
    if not feature_dir.is_dir():
        continue

    metrics_path = feature_dir / "metrics.json"
    if not metrics_path.exists():
        continue

    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows.append({
        "feature": feature_dir.name,
        "val_acc": m["val"]["acc"],
        "val_macro_f1": m["val"]["macro_f1"],
        "test_acc": m["test"]["acc"],
        "test_macro_f1": m["test"]["macro_f1"],
    })

if len(rows) == 0:
    raise RuntimeError(f"No metrics.json found under {OUT_ROOT}")

df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
out = OUT_ROOT / "summary.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print(df)
print("[OK] wrote ->", out)