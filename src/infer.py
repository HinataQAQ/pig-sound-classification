from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .features.gfcc import add_deltas as gfcc_delta
from .features.gfcc import compute_gfcc
from .features.mfcc import add_deltas as mfcc_delta
from .features.mfcc import compute_mfcc
from .models.bilstm import BiLSTMClassifier
from .preprocessing.denoise import spectral_subtraction
from .preprocessing.vad import detect_activity_regions
from .utils.audio_io import fix_length, load_wav


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--hop", type=float, default=None, help="sliding hop in seconds")
    ap.add_argument("--thr", type=float, default=None, help="cough probability threshold")
    ap.add_argument("--out_dir", type=str, default=None)
    return ap.parse_args()


def extract_features(y: np.ndarray, cfg: dict) -> np.ndarray:
    sr = int(cfg["data"]["sample_rate"])
    hop_len = int(round(0.010 * sr))
    win_len = int(round(0.025 * sr))
    hop_time = hop_len / float(sr)
    window_time = win_len / float(sr)

    feat_cfg = cfg["features"]
    mfcc_cfg = feat_cfg.get("mfcc", {})
    gfcc_cfg = feat_cfg.get("gfcc", {})

    mfcc = compute_mfcc(
        y,
        sr=sr,
        n_mfcc=int(mfcc_cfg.get("n_mfcc", 26)),
        n_mels=int(mfcc_cfg.get("n_mels", 64)),
        fmin=float(mfcc_cfg.get("fmin", 50)),
        fmax=float(mfcc_cfg.get("fmax", 8000)),
        hop_length=hop_len,
        win_length=win_len,
        center=False,
    )
    mfcc = mfcc_delta(mfcc, feat_cfg.get("include_delta", True), feat_cfg.get("include_delta_delta", False))

    gfcc = compute_gfcc(
        y,
        sr=sr,
        n_gfcc=int(gfcc_cfg.get("n_gfcc", 32)),
        n_filters=int(gfcc_cfg.get("n_filters", 64)),
        window_time=window_time,
        hop_time=hop_time,
        fmin=float(gfcc_cfg.get("fmin", 50)),
        fmax=float(gfcc_cfg.get("fmax", 8000)),
    )
    gfcc = gfcc_delta(gfcc, feat_cfg.get("include_delta", True), feat_cfg.get("include_delta_delta", False))

    T = min(mfcc.shape[0], gfcc.shape[0])
    feat = np.concatenate([mfcc[:T], gfcc[:T]], axis=1).astype(np.float32)

    if str(feat_cfg.get("cmvn", "utt")).lower() == "utt":
        mu = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / std
    return feat


def build_windows(total_len: int, seg_len: int, hop_len: int):
    if total_len <= seg_len:
        return [0]
    starts = list(range(0, total_len - seg_len + 1, hop_len))
    if starts[-1] != total_len - seg_len:
        starts.append(total_len - seg_len)
    return starts


def filter_windows_by_regions(starts, seg_len: int, regions, min_overlap_ratio: float = 0.10):
    if not regions:
        return starts
    out = []
    min_overlap = seg_len * float(min_overlap_ratio)
    for s in starts:
        e = s + seg_len
        keep = False
        for r in regions:
            overlap = max(0, min(e, int(r["end"])) - max(s, int(r["start"])))
            if overlap >= min_overlap:
                keep = True
                break
        if keep:
            out.append(s)
    return out or starts


def merge_events(frame_df: pd.DataFrame, seg_s: float, thr: float, merge_gap_s: float, min_event_s: float):
    starts = frame_df["start_s"].to_numpy()
    probs = frame_df["prob_cough"].to_numpy()
    is_pos = probs >= float(thr)

    events = []
    i = 0
    n = len(starts)

    while i < n:
        if not is_pos[i]:
            i += 1
            continue

        ev_start = float(starts[i])
        last_start = float(starts[i])
        ev_probs = [float(probs[i])]
        i += 1

        while i < n and is_pos[i] and (float(starts[i]) - last_start) <= float(merge_gap_s) + 1e-9:
            ev_probs.append(float(probs[i]))
            last_start = float(starts[i])
            i += 1

        ev_end = last_start + float(seg_s)
        dur = ev_end - ev_start
        if dur >= float(min_event_s):
            events.append(
                {
                    "start_s": ev_start,
                    "end_s": ev_end,
                    "duration_s": dur,
                    "max_prob": max(ev_probs),
                    "mean_prob": float(np.mean(ev_probs)),
                    "n_windows": len(ev_probs),
                }
            )

    return pd.DataFrame(events)


def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ckpt = torch.load(args.model, map_location="cpu")

    if "config" in ckpt and isinstance(ckpt["config"], dict):
        merged_cfg = ckpt["config"].copy()
        # 命令行 config 优先，用于小幅覆盖
        merged_cfg.update({k: v for k, v in cfg.items() if k not in merged_cfg})
        cfg = merged_cfg

    labels = [str(x).strip().lower() for x in ckpt.get("labels", cfg["data"]["labels"])]
    if "cough" not in labels:
        raise ValueError(f"labels has no 'cough': {labels}")
    cough_idx = labels.index("cough")

    sr = int(cfg["data"]["sample_rate"])
    seg_s = float(cfg["data"]["segment_seconds"])
    seg_len = int(round(sr * seg_s))

    infer_cfg = cfg.get("infer", {})
    hop_s = float(args.hop if args.hop is not None else infer_cfg.get("window_hop_seconds", 0.25))
    thr = float(args.thr if args.thr is not None else infer_cfg.get("prob_threshold", 0.65))
    out_dir = Path(args.out_dir or infer_cfg.get("out_dir", "infer_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    y, _ = load_wav(args.wav, sr=sr, mono=True)
    if bool(cfg.get("preprocess", {}).get("do_denoise", False)):
        y = spectral_subtraction(y, sr)

    vad_regions = []
    if bool(cfg.get("preprocess", {}).get("do_vad", False)):
        vad_info = detect_activity_regions(
            y=y,
            sr=sr,
            **cfg.get("preprocess", {}).get("vad", {}),
        )
        vad_regions = vad_info.get("regions", [])

    hop_len = max(1, int(round(hop_s * sr)))
    starts = build_windows(len(y), seg_len=seg_len, hop_len=hop_len)
    starts = filter_windows_by_regions(starts, seg_len=seg_len, regions=vad_regions, min_overlap_ratio=0.10)

    model = BiLSTMClassifier(
        input_size=int(ckpt["input_size"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_classes=len(labels),
        dropout=float(cfg["model"]["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    rows = []
    for start in starts:
        seg = y[start:start + seg_len]
        seg = fix_length(seg, seg_len)
        feat = extract_features(seg, cfg)
        x = torch.from_numpy(feat)[None, :, :]

        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(np.argmax(prob))
        rows.append(
            {
                "start_s": float(start / sr),
                "end_s": float((start + seg_len) / sr),
                "pred": labels[pred_idx],
                "prob_cough": float(prob[cough_idx]),
            }
        )

    frame_df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)
    event_df = merge_events(
        frame_df,
        seg_s=seg_s,
        thr=thr,
        merge_gap_s=float(infer_cfg.get("merge_gap_seconds", 0.25)),
        min_event_s=float(infer_cfg.get("min_event_seconds", 0.15)),
    )

    stem = Path(args.wav).stem
    frames_csv = out_dir / f"{stem}_frames.csv"
    events_csv = out_dir / f"{stem}_events.csv"
    frame_df.to_csv(frames_csv, index=False, encoding="utf-8-sig")
    event_df.to_csv(events_csv, index=False, encoding="utf-8-sig")

    if len(event_df) == 0:
        print(f"[INFO] no cough event above threshold {thr:.2f}")
    else:
        print(f"[INFO] detected {len(event_df)} cough event(s) above threshold {thr:.2f}")
        print(event_df.to_string(index=False))

    print("[OK] frames ->", frames_csv)
    print("[OK] events ->", events_csv)


if __name__ == "__main__":
    main()
