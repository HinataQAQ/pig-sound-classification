
import os
import sys
import math
import argparse
from pathlib import Path
from typing import Iterable, List

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running from repo root or tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.audio_io import load_wav, fix_length
from src.preprocessing.denoise import spectral_subtraction
from src.dataset_logmel import compute_logmel

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def iter_audio_files(inp: str) -> List[Path]:
    p = Path(inp)
    files: List[Path] = []

    if p.is_dir():
        for x in p.rglob("*"):
            if x.is_file() and x.suffix.lower() in AUDIO_EXTS:
                files.append(x)
        return sorted(files)

    if p.is_file() and p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = None
        for c in ("filepath", "path", "audio_path", "wav_path"):
            if c in df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"Cannot infer audio path column from CSV columns: {list(df.columns)}")
        for x in df[col].astype(str):
            xp = Path(x)
            if xp.exists():
                files.append(xp)
        return files

    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]

    raise FileNotFoundError(f"Input not found or unsupported: {inp}")


def center_crop_or_pad(y: np.ndarray, seg_len: int) -> np.ndarray:
    if len(y) < seg_len:
        return fix_length(y, seg_len)
    start = max(0, (len(y) - seg_len) // 2)
    return y[start:start + seg_len]


def maybe_cmvn(feat: np.ndarray, cmvn_mode: str) -> np.ndarray:
    if str(cmvn_mode).lower() != "utt":
        return feat
    mu = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-6
    return (feat - mu) / std


def sanitize(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = s
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def save_png(
    feat: np.ndarray,
    out_png: Path,
    title: str,
    cmap: str = "magma",
    dpi: int = 150,
    show_colorbar: bool = True,
) -> None:
    # feat shape: (T, F)
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(feat.T, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="audio file, folder, or manifest CSV")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--full_clip", action="store_true", help="use full clip instead of center-cropping to config segment_seconds")
    ap.add_argument("--no_denoise", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cmap", type=str, default="magma")
    ap.add_argument("--no_colorbar", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    sr = int(cfg["data"].get("sample_rate", 32000))
    seg_s = float(cfg["data"].get("segment_seconds", 1.0))
    seg_len = int(round(sr * seg_s))

    lm = cfg.get("logmel", {})
    n_mels = int(lm.get("n_mels", 64))
    fmin = float(lm.get("fmin", 50))
    fmax = float(lm.get("fmax", 8000))
    hop_ms = float(lm.get("hop_ms", 10.0))
    win_ms = float(lm.get("win_ms", 25.0))
    n_fft = int(lm.get("n_fft", 1024))
    ref_mode = str(lm.get("ref_mode", "fixed"))
    db_ref = float(lm.get("db_ref", 1.0))
    top_db = float(lm.get("top_db", 80.0))
    cmvn_mode = str(lm.get("cmvn", "none"))
    do_denoise = (not args.no_denoise) and bool(cfg.get("preprocess", {}).get("do_denoise", False))

    files = iter_audio_files(args.input)
    if args.limit > 0:
        files = files[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, wav_path in enumerate(files, start=1):
        y, _ = load_wav(str(wav_path), sr=sr, mono=True)
        if do_denoise:
            y = spectral_subtraction(y, sr)

        dur_s_raw = len(y) / sr
        if not args.full_clip:
            y = center_crop_or_pad(y, seg_len)

        feat = compute_logmel(
            y=y,
            sr=sr,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            hop_ms=hop_ms,
            win_ms=win_ms,
            n_fft=n_fft,
            ref_mode=ref_mode,
            db_ref=db_ref,
            top_db=top_db,
        )
        feat = maybe_cmvn(feat, cmvn_mode)

        stem = sanitize(wav_path.stem)
        out_png = out_dir / f"{i:04d}__{stem}.png"
        title = f"{wav_path.name} | raw={dur_s_raw:.2f}s"
        save_png(
            feat=feat,
            out_png=out_png,
            title=title,
            cmap=args.cmap,
            dpi=args.dpi,
            show_colorbar=(not args.no_colorbar),
        )

        rows.append(
            {
                "audio_path": str(wav_path.resolve()),
                "png_path": str(out_png.resolve()),
                "raw_duration_s": round(dur_s_raw, 4),
                "used_full_clip": bool(args.full_clip),
                "frames": int(feat.shape[0]),
                "mel_bins": int(feat.shape[1]),
            }
        )

        if i % 20 == 0:
            print(f"[INFO] exported {i}/{len(files)}")

    pd.DataFrame(rows).to_csv(out_dir / "_index.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {len(rows)} pngs to: {out_dir}")


if __name__ == "__main__":
    main()
