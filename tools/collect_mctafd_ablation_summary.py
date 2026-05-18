from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports"

rows = []

patterns = [
    "ablation_*",
    "seed_*",
    "logmel_param_*",
    "large_*",
]

for pat in patterns:
    for d in REPORT_ROOT.glob(pat):
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

            "n_mels": m.get("n_mels"),
            "n_mfcc": m.get("n_mfcc"),
            "fmin": m.get("fmin"),
            "fmax": m.get("fmax"),
            "n_fft": m.get("n_fft"),
            "hop_length": m.get("hop_length"),
            "win_length": m.get("win_length"),
            "sr": m.get("sr"),
            "dur_s": m.get("dur_s"),

            "best_epoch": m.get("best_epoch"),
            "best_val_macro_f1": m.get("best_val_macro_f1"),
            "test_acc": m.get("test_acc"),
            "test_macro_f1": m.get("test_macro_f1"),
        })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise RuntimeError("No summary.json found under reports/")

df = df.sort_values(["experiment"])
out = REPORT_ROOT / "mctafd_ablation_summary.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print(df.to_string(index=False))
print("[OK] wrote ->", out)