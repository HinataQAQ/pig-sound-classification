#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess audio: resample -> (optional) denoise -> segment fixed-length clips.
Usage (Windows path example):
  python tools/preprocess.py --in_dir data/raw/soundwel --out_dir data/processed/soundwel --sr 16000 --clip_sec 2.0 --denoise
Directory assumptions:
  - Input:  data/raw/<dataset>/<label>/*.wav
  - Output: data/processed/<dataset>/<label>/*_segXX.wav
"""
import os, argparse, math
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa

try:
    import noisereduce as nr
    HAVE_NR = True
except Exception:
    HAVE_NR = False

def segment_and_save(y, sr, out_path_base: Path, clip_sec: float, min_keep_ratio: float = 0.5):
    seg_len = int(sr * clip_sec)
    n_seg = max(1, math.floor(len(y) / seg_len))
    saved = 0
    for i in range(n_seg + 1):
        start = i * seg_len
        end = start + seg_len
        if start >= len(y):
            break
        clip = y[start:end]
        if len(clip) < int(min_keep_ratio * seg_len):
            break
        if len(clip) < seg_len:
            clip = np.pad(clip, (0, seg_len - len(clip)))
        out_path = out_path_base.with_name(out_path_base.stem + f"_seg{i:02d}" + out_path_base.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path.as_posix(), clip, sr)
        saved += 1
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="raw audio root, e.g., data/raw/soundwel")
    ap.add_argument("--out_dir", required=True, help="processed audio root, e.g., data/processed/soundwel")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--clip_sec", type=float, default=2.0)
    ap.add_argument("--denoise", action="store_true", help="apply spectral gating denoise (noisereduce)")
    args = ap.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    assert in_root.exists(), f"Input dir not found: {in_root}"

    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = [p for p in in_root.rglob("*") if p.suffix.lower() in exts]
    total_saved = 0

    for p in files:
        # label = parent folder name
        label = p.parent.name
        y, s = librosa.load(p.as_posix(), sr=args.sr, mono=True)
        if args.denoise and HAVE_NR:
            # spectral gating denoise
            y = nr.reduce_noise(y=y, sr=args.sr)
        # output path preserves label folder
        rel = p.relative_to(in_root)
        out_path_base = (out_root / rel).with_suffix(".wav")
        out_path_base = out_path_base.parent / out_path_base.name  # ensure label folder exists
        saved = segment_and_save(y, args.sr, out_path_base, args.clip_sec)
        total_saved += saved
        print(f"[OK] {p.name} -> {saved} clips")
    print(f"Done. Total clips saved: {total_saved}; out_dir={out_root}")

if __name__ == "__main__":
    main()
