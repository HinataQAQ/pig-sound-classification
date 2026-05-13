import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset_logmel import compute_logmel
from src.models.crnn import CRNNClassifier
from src.preprocessing.denoise import spectral_subtraction
from src.utils.audio_io import fix_length, load_wav


def build_model(cfg: dict, labels):
    mcfg = cfg.get('model', {})
    return CRNNClassifier(
        n_mels=int(cfg.get('logmel', {}).get('n_mels', 64)),
        num_classes=len(labels),
        cnn_channels=tuple(mcfg.get('cnn_channels', [16, 32, 64])),
        rnn_hidden=int(mcfg.get('rnn_hidden', 128)),
        rnn_layers=int(mcfg.get('rnn_layers', 2)),
        dropout=float(mcfg.get('dropout', 0.0)),
    )


def apply_cmvn_if_needed(feat: np.ndarray, cmvn_mode: str) -> np.ndarray:
    mode = str(cmvn_mode).lower()
    if mode != 'utt':
        return feat
    mu = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-6
    return (feat - mu) / std


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    xpad = np.pad(x.astype(np.float32), (pad_left, pad_right), mode='edge')
    return np.convolve(xpad, kernel, mode='valid').astype(np.float32)


def merge_events(
    frame_df: pd.DataFrame,
    seg_s: float,
    thr_high: float,
    thr_low: float | None = None,
    merge_gap_s: float = 0.20,
    min_event_s: float = 0.20,
) -> pd.DataFrame:
    starts = frame_df['start_s'].to_numpy(dtype=np.float32)
    probs = frame_df['prob_cough_smooth'].to_numpy(dtype=np.float32) if 'prob_cough_smooth' in frame_df.columns else frame_df['prob_cough'].to_numpy(dtype=np.float32)

    if thr_low is None:
        thr_low = thr_high

    events = []
    i = 0
    n = len(starts)
    while i < n:
        if probs[i] < thr_high:
            i += 1
            continue

        ev_start = float(starts[i])
        last_start = float(starts[i])
        ev_probs = [float(probs[i])]
        i += 1

        while i < n:
            close_enough = (float(starts[i]) - last_start) <= float(merge_gap_s) + 1e-9
            still_active = probs[i] >= float(thr_low)
            if not (close_enough and still_active):
                break
            ev_probs.append(float(probs[i]))
            last_start = float(starts[i])
            i += 1

        ev_end = last_start + float(seg_s)
        dur = ev_end - ev_start
        if dur >= float(min_event_s):
            events.append(
                {
                    'start_s': ev_start,
                    'end_s': ev_end,
                    'duration_s': dur,
                    'max_prob': float(np.max(ev_probs)),
                    'mean_prob': float(np.mean(ev_probs)),
                    'n_frames': int(len(ev_probs)),
                }
            )

    return pd.DataFrame(events)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out_prefix', default='det')
    ap.add_argument('--thr', type=float, default=0.70)
    ap.add_argument('--thr_low', type=float, default=None, help='optional hysteresis low threshold; default == --thr')
    ap.add_argument('--hop', type=float, default=0.10)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--no_denoise', action='store_true')
    ap.add_argument('--smooth', type=int, default=1, help='moving-average window on frame probs, in frames')
    ap.add_argument('--merge_gap', type=float, default=0.20)
    ap.add_argument('--min_event', type=float, default=0.20)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_file = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] device =', device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', cfg_file)
    if not isinstance(cfg, dict):
        cfg = cfg_file

    labels = [str(x).strip().lower() for x in ckpt.get('labels', cfg['data']['labels'])]
    if 'cough' not in labels:
        raise ValueError(f"labels has no 'cough': {labels}")
    cough_idx = labels.index('cough')

    model = build_model(cfg, labels).to(device)
    incompatible = model.load_state_dict(ckpt['state_dict'], strict=True)
    print('[INFO] load_state_dict =', incompatible)
    model.eval()

    sr = int(cfg['data'].get('sample_rate', 32000))
    seg_s = float(cfg['data'].get('segment_seconds', 1.0))
    seg_len = int(round(sr * seg_s))
    hop_s = float(args.hop)
    hop_len = max(1, int(round(sr * hop_s)))

    y, _ = load_wav(args.wav, sr=sr, mono=True)
    if (not args.no_denoise) and bool(cfg.get('preprocess', {}).get('do_denoise', False)):
        y = spectral_subtraction(y, sr)
    if len(y) < seg_len:
        y = fix_length(y, seg_len)

    lm = cfg.get('logmel', {})
    ref_mode = str(lm.get('ref_mode', 'fixed'))
    db_ref = float(lm.get('db_ref', 1.0))
    top_db = float(lm.get('top_db', 80.0))
    cmvn_mode = str(lm.get('cmvn', 'none'))

    total = len(y)
    starts = list(range(0, max(1, total - seg_len + 1), hop_len))
    tail = max(0, total - seg_len)
    if not starts or starts[-1] != tail:
        starts.append(tail)

    rows = []
    feats = []
    meta = []

    def flush_batch():
        nonlocal feats, meta, rows
        if not feats:
            return
        x = torch.from_numpy(np.stack(feats, axis=0)).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy()
        for (start_s, end_s, idx), p in zip(meta, probs):
            rows.append(
                {
                    'frame_idx': int(idx),
                    'start_s': float(start_s),
                    'end_s': float(end_s),
                    'prob_cough': float(p[cough_idx]),
                }
            )
        feats = []
        meta = []

    for idx, s in enumerate(tqdm(starts, desc='[scan]')):
        seg = y[s:s + seg_len]
        seg = fix_length(seg, seg_len)
        feat = compute_logmel(
            y=seg,
            sr=sr,
            n_mels=int(lm.get('n_mels', 64)),
            fmin=float(lm.get('fmin', 50)),
            fmax=float(lm.get('fmax', 8000)),
            hop_ms=float(lm.get('hop_ms', 10.0)),
            win_ms=float(lm.get('win_ms', 25.0)),
            n_fft=int(lm.get('n_fft', 1024)),
            ref_mode=ref_mode,
            db_ref=db_ref,
            top_db=top_db,
        )
        feat = apply_cmvn_if_needed(feat, cmvn_mode)
        feats.append(feat.astype(np.float32))
        meta.append((s / sr, (s + seg_len) / sr, idx))
        if len(feats) >= int(args.batch_size):
            flush_batch()
    flush_batch()

    frame_df = pd.DataFrame(rows).sort_values('start_s').reset_index(drop=True)
    frame_df['prob_cough_smooth'] = moving_average(frame_df['prob_cough'].to_numpy(), int(args.smooth))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_frames = f'{args.out_prefix}_frames.csv'
    frame_df.to_csv(out_frames, index=False, encoding='utf-8-sig')
    print('[OK] wrote frames ->', os.path.abspath(out_frames))

    event_df = merge_events(
        frame_df,
        seg_s=seg_s,
        thr_high=float(args.thr),
        thr_low=args.thr_low,
        merge_gap_s=float(args.merge_gap),
        min_event_s=float(args.min_event),
    )
    out_events = f'{args.out_prefix}_events.csv'
    event_df.to_csv(out_events, index=False, encoding='utf-8-sig')
    print('[OK] wrote events ->', os.path.abspath(out_events))
    print(f'[DONE] frames={len(frame_df)} events={len(event_df)} thr={args.thr:.3f} hop={hop_s}s seg={seg_s}s')


if __name__ == '__main__':
    main()
