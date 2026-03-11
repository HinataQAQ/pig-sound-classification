# tools/detect_long_cough.py
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 你之前遇到过 OMP Error #15

# ✅ 关键：保证从任何工作目录运行都能 import src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os, sys

# 解决你之前遇到过的 OMP Error #15（不想每次崩就加这个）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 保证 tools/*.py 也能 import src.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.audio_io import load_wav, fix_length
from src.preprocessing.denoise import spectral_subtraction
from src.models.crnn import CRNNClassifier


def compute_logmel(
    y, sr,
    n_mels=64, fmin=50, fmax=8000,
    hop_ms=10.0, win_ms=25.0, n_fft=1024,
    top_db=80.0
):
    """
    和你 eval_crnn_logmel 的风格一致：logmel in dB, shape (T, F), max≈0, min≈-top_db
    """
    import librosa
    hop_length = int(sr * hop_ms / 1000.0)
    win_length = int(sr * win_ms / 1000.0)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
        power=2.0, center=False
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=float(top_db)).T
    return logmel.astype(np.float32)


def cmvn(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + eps
    return (feat - mu) / std


def merge_events(frame_df: pd.DataFrame, seg_s: float, hop_s: float, thr: float,
                 merge_gap_s: float = 0.20, min_event_s: float = 0.20) -> pd.DataFrame:
    """
    把 frame-level (start_s, prob_cough) 合并成 event-level
    - 只要相邻“阳性窗”的 start 间隔 <= merge_gap_s 就合并
    - event_end = last_start + seg_s
    """
    starts = frame_df["start_s"].to_numpy()
    probs = frame_df["prob_cough"].to_numpy()
    is_pos = probs >= thr

    events = []
    i = 0
    n = len(starts)

    while i < n:
        if not is_pos[i]:
            i += 1
            continue

        # start an event
        ev_start = starts[i]
        ev_probs = [probs[i]]
        last_start = starts[i]
        i += 1

        while i < n and is_pos[i] and (starts[i] - last_start) <= merge_gap_s + 1e-9:
            ev_probs.append(probs[i])
            last_start = starts[i]
            i += 1

        ev_end = last_start + seg_s
        dur = ev_end - ev_start
        if dur >= min_event_s:
            events.append({
                "start_s": float(ev_start),
                "end_s": float(ev_end),
                "duration_s": float(dur),
                "max_prob": float(np.max(ev_probs)),
                "mean_prob": float(np.mean(ev_probs)),
                "n_frames": int(len(ev_probs)),
            })

    return pd.DataFrame(events)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="path to long wav")
    ap.add_argument("--ckpt", required=True, help="trained ckpt .pt")
    ap.add_argument("--config", required=True, help="yaml config used for model/logmel")
    ap.add_argument("--out_prefix", default="det", help="prefix for output csv")
    ap.add_argument("--thr", type=float, default=0.70, help="prob threshold for cough")
    ap.add_argument("--hop", type=float, default=0.10, help="hop seconds for sliding window")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_denoise", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    labels = ckpt.get("labels", cfg["data"]["labels"])
    labels = [str(x).strip().lower() for x in labels]
    if "cough" not in labels:
        raise ValueError(f"[ERR] labels has no 'cough': {labels}")
    cough_idx = labels.index("cough")

    # model
    n_mels = int(cfg.get("logmel", {}).get("n_mels", 64))
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})  # 你 train_crnn.py 里保存进去的那个 cfg

    labels = ckpt.get("labels", cfg.get("data", {}).get("labels", ["cough", "other"]))
    logmel_cfg = cfg.get("logmel", {})
    model_cfg = cfg.get("model", {})

    # 这几个值必须和训练时一致，否则就会 size mismatch
    model = CRNNClassifier(
        n_mels=int(logmel_cfg.get("n_mels", 64)),
        num_classes=len(labels),
        cnn_channels=tuple(model_cfg.get("cnn_channels", (16, 32, 64))),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 128)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("[DEBUG] missing keys =", missing)
    print("[DEBUG] unexpected keys =", unexpected)

    model.eval()

    # audio
    sr = int(cfg["data"].get("sample_rate", 32000))
    seg_s = float(cfg["data"].get("segment_seconds", 1.0))
    seg_len = int(sr * seg_s)
    hop_s = float(args.hop)
    hop_len = max(1, int(sr * hop_s))

    y, _ = load_wav(args.wav, sr=sr, mono=True)
    if (not args.no_denoise) and bool(cfg.get("preprocess", {}).get("do_denoise", False)):
        y = spectral_subtraction(y, sr)

    if len(y) < seg_len:
        y = fix_length(y, seg_len)

    # logmel cfg
    lm = cfg.get("logmel", {})
    fmin = float(lm.get("fmin", 50))
    fmax = float(lm.get("fmax", 8000))
    hop_ms = float(lm.get("hop_ms", 10.0))
    win_ms = float(lm.get("win_ms", 25.0))
    n_fft = int(lm.get("n_fft", 1024))
    top_db = float(lm.get("top_db", 80.0))

    rows = []
    feats = []
    meta = []

    def flush_batch():
        nonlocal feats, meta
        if not feats:
            return
        x = torch.from_numpy(np.stack(feats, axis=0)).to(device)  # (B,T,F)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        for (start_s,), p in zip(meta, probs):
            rows.append({
                "start_s": float(start_s),
                "prob_cough": float(p[cough_idx]),
            })
        feats = []
        meta = []

    total = len(y)
    starts = range(0, total - seg_len + 1, hop_len)
    for s in tqdm(starts, desc="[scan]"):
        seg = y[s:s + seg_len]
        feat = compute_logmel(
            seg, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
            hop_ms=hop_ms, win_ms=win_ms, n_fft=n_fft, top_db=top_db
        )
        feat = cmvn(feat)  # 和你训练一致：每窗 CMVN
        feats.append(feat)
        meta.append((s / sr,))

        if len(feats) >= args.batch_size:
            flush_batch()

    flush_batch()

    frame_df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)

    out_frames = f"{args.out_prefix}_frames.csv"
    frame_df.to_csv(out_frames, index=False, encoding="utf-8-sig")
    print("[OK] wrote frames ->", os.path.abspath(out_frames))

    event_df = merge_events(frame_df, seg_s=seg_s, hop_s=hop_s, thr=float(args.thr))
    out_events = f"{args.out_prefix}_events.csv"
    event_df.to_csv(out_events, index=False, encoding="utf-8-sig")
    print("[OK] wrote events ->", os.path.abspath(out_events))

    print(f"[DONE] frames={len(frame_df)} events={len(event_df)} thr={args.thr} hop={hop_s}s seg={seg_s}s")


if __name__ == "__main__":
    main()
