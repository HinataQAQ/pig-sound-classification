# tools/make_aswine_cough_silence_meta.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EVENT_COLS = [
    "Limpeza de Baia",
    "Fala Humana",
    "Alimentação de Baias",
    "OutrosH",
    "EstresseDisputas",
    "TosseEspirro",
    "OutrosA",
]

def to_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("true", "1", "t", "yes", "y")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=r"data\external\aswine\meta\1s_pruned\aswine_1s_pruned_full.csv")
    ap.add_argument("--out", type=str, default=r"data\external\aswine\meta\1s_pruned\aswine_pure_cough_silence_other.csv")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max_silence", type=int, default=0, help="0 means match cough count; >0 means cap silence count")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    missing = [c for c in ["audio_file_path", "offset", "duration"] + EVENT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nGot columns: {list(df.columns)[:30]} ...")

    # normalize bool cols
    for c in EVENT_COLS:
        df[c] = df[c].map(to_bool)

    other_event_cols = [c for c in EVENT_COLS if c != "TosseEspirro"]

    # pure cough: TosseEspirro=True and all other events False
    pure_cough = df[(df["TosseEspirro"] == True)]
    for c in other_event_cols:
        pure_cough = pure_cough[pure_cough[c] == False]

    # pure silence/background: all events False (including TosseEspirro)
    pure_silence = df.copy()
    for c in EVENT_COLS:
        pure_silence = pure_silence[pure_silence[c] == False]

    n_cough = len(pure_cough)
    n_sil = len(pure_silence)

    rng = np.random.default_rng(args.seed)
    take_sil = n_cough if args.max_silence == 0 else min(args.max_silence, n_sil)
    if take_sil > n_sil:
        raise ValueError(f"Not enough silence rows: need {take_sil}, have {n_sil}")

    sil_idx = rng.choice(pure_silence.index.to_numpy(), size=take_sil, replace=False)
    sil_take = pure_silence.loc[sil_idx].copy()

    pure_cough_out = pure_cough.copy()
    pure_cough_out["label"] = "cough"
    sil_take["label"] = "other"

    out = pd.concat([pure_cough_out, sil_take], axis=0, ignore_index=True)
    out = out[["audio_file_path", "offset", "duration", "label"]].copy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")

    print("pure_cough =", n_cough)
    print("pure_silence =", n_sil)
    print("saved =", len(out), "->", out_path)
    print(out["label"].value_counts())

if __name__ == "__main__":
    main()
