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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest = ROOT / args.manifest
    df = pd.read_csv(manifest)
    path_col = infer_path_col(df)

    durations = []

    missing = 0

    for _, r in df.iterrows():
        p = ROOT / str(r[path_col]).replace("\\", "/")

        if not p.exists():
            missing += 1
            continue

        info = sf.info(str(p))
        dur = float(info.frames) / float(info.samplerate)
        durations.append(dur)

    arr = np.array(durations, dtype=np.float32)

    print(f"[MANIFEST] {manifest}")
    print(f"n_files       = {len(arr)}")
    print(f"missing       = {missing}")

    if len(arr) == 0:
        return

    print(f"min_s         = {arr.min():.4f}")
    print(f"p25_s         = {np.percentile(arr, 25):.4f}")
    print(f"median_s      = {np.median(arr):.4f}")
    print(f"mean_s        = {arr.mean():.4f}")
    print(f"p75_s         = {np.percentile(arr, 75):.4f}")
    print(f"max_s         = {arr.max():.4f}")
    print(f"<=1.05s       = {(arr <= 1.05).sum()} / {len(arr)}")
    print(f">1.5s         = {(arr > 1.5).sum()} / {len(arr)}")
    print(f">2.0s         = {(arr > 2.0).sum()} / {len(arr)}")
    print(f">3.0s         = {(arr > 3.0).sum()} / {len(arr)}")


if __name__ == "__main__":
    main()