# tools/make_aswine_pure_activity_csv.py
import pandas as pd
from pathlib import Path

CSV_IN = Path(r"data/external/aswine/meta/1s_pruned/aswine_1s_pruned_full.csv")
CSV_OUT = Path(r"data/external/aswine/meta/1s_pruned/aswine_pure_cough_activity_other.csv")

BOOL_COLS = [
    "Limpeza de Baia",
    "Fala Humana",
    "Alimentação de Baias",
    "OutrosH",
    "EstresseDisputas",
    "TosseEspirro",
    "OutrosA",
]

def to_bool(x):
    s = str(x).strip().lower()
    return s in ("true", "1", "t", "yes", "y")

df = pd.read_csv(CSV_IN)

# 把这些事件列统一转成 bool
for c in BOOL_COLS:
    df[c] = df[c].apply(to_bool)

is_cough = df["TosseEspirro"]
other_cols = [c for c in BOOL_COLS if c != "TosseEspirro"]

pure_cough = is_cough & (~df[other_cols].any(axis=1))
activity_other = (~is_cough) & (df[other_cols].any(axis=1))

use = df.loc[pure_cough | activity_other, ["audio_file_path", "offset", "duration", "TosseEspirro"]].copy()
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
use.to_csv(CSV_OUT, index=False)

print("pure_cough =", int(pure_cough.sum()))
print("activity_other =", int(activity_other.sum()))
print("wrote:", CSV_OUT)
