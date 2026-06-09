from pathlib import Path
import argparse
import pandas as pd
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def infer_path_col(df):
    for c in ["path", "filepath", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column from columns={list(df.columns)}")


def infer_label_col(df):
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column from columns={list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest = ROOT / args.manifest
    df = pd.read_csv(manifest)

    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    rows = []

    for _, r in df.iterrows():
        p = ROOT / str(r[path_col]).replace("\\", "/")
        lab = str(r[label_col]).strip().lower()

        if not p.exists():
            continue

        info = sf.info(str(p))
        dur = float(info.frames) / float(info.samplerate)

        rows.append({"label": lab, "duration_s": dur})

    out = pd.DataFrame(rows)

    summary = (
        out.groupby("label")["duration_s"]
        .agg(
            n="count",
            min_s="min",
            p25_s=lambda x: np.percentile(x, 25),
            median_s="median",
            mean_s="mean",
            p75_s=lambda x: np.percentile(x, 75),
            max_s="max",
        )
        .reset_index()
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()