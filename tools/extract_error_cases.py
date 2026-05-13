import pandas as pd
from pathlib import Path

IN_CSV = Path(r"C:\py\pigsound\pig-sound-classification\reports\transfer_from_korea_clean_test_pred.csv")
OUT_DIR = Path(r"C:\py\pigsound\pig-sound-classification\reports\error_analysis_transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_CSV)
print("columns =", list(df.columns))

# 自动找列名
path_col = None
for c in ["filepath", "path", "audio_path", "wav_path"]:
    if c in df.columns:
        path_col = c
        break

true_col = None
for c in ["true", "label", "y_true", "true_label"]:
    if c in df.columns:
        true_col = c
        break

pred_col = None
for c in ["pred", "y_pred", "pred_label"]:
    if c in df.columns:
        pred_col = c
        break

if true_col is None or pred_col is None:
    raise RuntimeError(f"Cannot infer true/pred columns. columns={list(df.columns)}")

err = df[df[true_col].astype(str) != df[pred_col].astype(str)].copy()

fn = err[(err[true_col].astype(str).str.lower() == "cough") & (err[pred_col].astype(str).str.lower() == "other")]
fp = err[(err[true_col].astype(str).str.lower() == "other") & (err[pred_col].astype(str).str.lower() == "cough")]

err.to_csv(OUT_DIR / "all_errors.csv", index=False, encoding="utf-8-sig")
fn.to_csv(OUT_DIR / "false_negative_cough_as_other.csv", index=False, encoding="utf-8-sig")
fp.to_csv(OUT_DIR / "false_positive_other_as_cough.csv", index=False, encoding="utf-8-sig")

print("all errors =", len(err))
print("false negatives cough->other =", len(fn))
print("false positives other->cough =", len(fp))
print("wrote ->", OUT_DIR)