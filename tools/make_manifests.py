#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan processed audio directory and create train/val/test manifests.
Usage:
  python tools/make_manifests.py --audio_root data/processed/soundwel --out_csv data/manifests --val_ratio 0.2 --test_ratio 0.2
It expects subfolders as class labels:
  data/processed/soundwel/<label>/*.wav
Manifests have 2 columns: filepath,label
"""
import os, argparse, random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True, help="processed audio root, e.g., data/processed/soundwel")
    ap.add_argument("--out_csv", required=True, help="output dir for csv, e.g., data/manifests")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    out_dir = Path(args.out_csv); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels = []
    for label_dir in sorted([p for p in audio_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        labels.append(label)
        for wav in label_dir.rglob("*.wav"):
            rows.append({"filepath": wav.as_posix(), "label": label})
    assert rows, f"No wav files found under {audio_root}"
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # split test first, then val
    df_trainval, df_test = train_test_split(df, test_size=args.test_ratio, stratify=df["label"], random_state=args.seed)
    val_size = args.val_ratio / (1 - args.test_ratio)
    df_train, df_val = train_test_split(df_trainval, test_size=val_size, stratify=df_trainval["label"], random_state=args.seed)

    df_train.to_csv((out_dir / "train.csv").as_posix(), index=False)
    df_val.to_csv((out_dir / "val.csv").as_posix(), index=False)
    df_test.to_csv((out_dir / "test.csv").as_posix(), index=False)

    print(f"[OK] Wrote: {out_dir/'train.csv'}, {out_dir/'val.csv'}, {out_dir/'test.csv'}")
    print("Labels (copy these into config/config.yaml):", sorted(labels))

if __name__ == "__main__":
    main()
