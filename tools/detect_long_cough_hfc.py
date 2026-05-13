import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
import librosa

from src.utils.audio_io import load_wav, fix_length
from src.preprocessing.denoise import spectral_subtraction
from src.models.crnn import CRNNClassifier


def robust_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)


def sigmoid01(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def smooth_1d(x: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    ys = np.convolve(xp, kernel, mode="valid")
    return ys.astype(np.float32)


def compute_logmel_cfg(y: np.ndarray, sr: int, logmel_cfg: dict) -> np.ndarray:
    n_mels = int(logmel_cfg.get("n_mels", 64))
    fmin = float(logmel_cfg.get("fmin", 50))
    fmax = float(logmel_cfg.get("fmax", 8000))
    hop_ms = float(logmel_cfg.get("hop_ms", 10.0))
    win_ms = float(logmel_cfg.get("win_ms", 25.0))
    n_fft = int(logmel_cfg.get("n_fft", 1024))
    top_db = float(logmel_cfg.get("top_db", 80.0))
    ref_mode = str(logmel_cfg.get("ref_mode", "fixed")).lower()
    db_ref = float(logmel_cfg.get("db_ref", 1.0))
    cmvn_mode = str(logmel_cfg.get("cmvn", "none")).lower()

    hop_length = int(sr * hop_ms / 1000.0)
    win_length = int(sr * win_ms / 1000.0)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=False,
    )

    if ref_mode == "fixed":
        logmel = librosa.power_to_db(mel, ref=db_ref, top_db=top_db)
    else:
        logmel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)

    feat = logmel.T.astype(np.float32)

    if cmvn_mode == "utt":
        mu = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / std

    return feat


def compute_aux_metrics(
    y: np.ndarray,
    sr: int,
    logmel_cfg: dict,
    band_lo: float = 500.0,
    band_hi: float = 2000.0,
):
    hop_ms = float(logmel_cfg.get("hop_ms", 10.0))
    win_ms = float(logmel_cfg.get("win_ms", 25.0))
    n_fft = int(logmel_cfg.get("n_fft", 1024))

    hop_length = int(sr * hop_ms / 1000.0)
    win_length = int(sr * win_ms / 1000.0)

    S = np.abs(
        librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
        )
    ).astype(np.float32)

    if S.size == 0 or S.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0.0

    bin_idx = (np.arange(S.shape[0], dtype=np.float32) + 1.0)[:, None]
    hfc_frames = (bin_idx * S).sum(axis=0)

    if S.shape[1] >= 2:
        flux_frames = np.maximum(np.diff(S, axis=1), 0.0).sum(axis=0)
        flux = float(np.percentile(flux_frames, 95))
    else:
        flux = 0.0

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= band_lo) & (freqs <= band_hi)
    band_energy = float(S[band_mask].sum())
    total_energy = float(S.sum()) + 1e-8
    band_ratio = band_energy / total_energy

    flatness = float(librosa.feature.spectral_flatness(S=S + 1e-8).mean())
    hfc = float(np.percentile(hfc_frames, 95))

    return hfc, flux, band_ratio, flatness


def build_starts(total_len: int, seg_len: int, hop_len: int):
    if total_len <= seg_len:
        return [0]
    starts = list(range(0, total_len - seg_len + 1, hop_len))
    if starts[-1] != total_len - seg_len:
        starts.append(total_len - seg_len)
    return starts


def merge_events_hysteresis(
    frame_df: pd.DataFrame,
    seg_s: float,
    high_thr: float,
    low_thr: float,
    merge_gap_s: float = 0.08,
    min_event_s: float = 0.15,
    score_col: str = "score_smooth",
) -> pd.DataFrame:
    starts = frame_df["start_s"].to_numpy(dtype=np.float32)
    scores = frame_df[score_col].to_numpy(dtype=np.float32)

    raw_events = []
    i = 0
    n = len(starts)

    while i < n:
        if scores[i] < high_thr:
            i += 1
            continue

        ev_start = float(starts[i])
        ev_scores = [float(scores[i])]
        last_start = float(starts[i])
        i += 1

        while i < n and scores[i] >= low_thr:
            ev_scores.append(float(scores[i]))
            last_start = float(starts[i])
            i += 1

        ev_end = last_start + float(seg_s)
        dur = ev_end - ev_start

        if dur >= float(min_event_s):
            raw_events.append(
                {
                    "start_s": ev_start,
                    "end_s": ev_end,
                    "duration_s": dur,
                    "max_score": float(np.max(ev_scores)),
                    "mean_score": float(np.mean(ev_scores)),
                    "n_frames": int(len(ev_scores)),
                }
            )

    if not raw_events:
        return pd.DataFrame(columns=["start_s", "end_s", "duration_s", "max_score", "mean_score", "n_frames"])

    merged = [raw_events[0]]
    for ev in raw_events[1:]:
        if ev["start_s"] - merged[-1]["end_s"] <= float(merge_gap_s):
            merged[-1]["end_s"] = ev["end_s"]
            merged[-1]["duration_s"] = merged[-1]["end_s"] - merged[-1]["start_s"]
            merged[-1]["max_score"] = max(merged[-1]["max_score"], ev["max_score"])
            merged[-1]["mean_score"] = float((merged[-1]["mean_score"] + ev["mean_score"]) / 2.0)
            merged[-1]["n_frames"] += ev["n_frames"]
        else:
            merged.append(ev)

    return pd.DataFrame(merged)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_prefix", default="det_hfc")
    ap.add_argument("--thr", type=float, default=0.58)
    ap.add_argument("--thr_low", type=float, default=0.48)
    ap.add_argument("--hop", type=float, default=0.10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--merge_gap", type=float, default=0.08)
    ap.add_argument("--min_event", type=float, default=0.15)
    ap.add_argument("--band_lo", type=float, default=500.0)
    ap.add_argument("--band_hi", type=float, default=2000.0)
    ap.add_argument("--no_denoise", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    yaml_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", yaml_cfg)

    labels = ckpt.get("labels", cfg.get("data", {}).get("labels", ["cough", "other"]))
    labels = [str(x).strip().lower() for x in labels]
    if "cough" not in labels:
        raise ValueError(f"[ERR] labels has no 'cough': {labels}")
    cough_idx = labels.index("cough")

    logmel_cfg = cfg.get("logmel", {})
    model_cfg = cfg.get("model", {})

    model = CRNNClassifier(
        n_mels=int(logmel_cfg.get("n_mels", 64)),
        num_classes=len(labels),
        cnn_channels=tuple(model_cfg.get("cnn_channels", (16, 32, 64))),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 128)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    msg = model.load_state_dict(ckpt["state_dict"], strict=True)
    print("[INFO] load_state_dict =", msg)
    model.eval()

    sr = int(cfg["data"].get("sample_rate", 32000))
    seg_s = float(cfg["data"].get("segment_seconds", 1.0))
    seg_len = int(sr * seg_s)
    hop_len = max(1, int(sr * float(args.hop)))

    y, _ = load_wav(args.wav, sr=sr, mono=True)
    if (not args.no_denoise) and bool(cfg.get("preprocess", {}).get("do_denoise", False)):
        y = spectral_subtraction(y, sr)

    if len(y) < seg_len:
        y = fix_length(y, seg_len)

    starts = build_starts(len(y), seg_len=seg_len, hop_len=hop_len)

    rows = []
    feats = []
    meta = []

    def flush_batch():
        nonlocal feats, meta, rows
        if not feats:
            return
        x = torch.from_numpy(np.stack(feats, axis=0)).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        for item, p in zip(meta, probs):
            rows.append(
                {
                    "frame_idx": int(item["frame_idx"]),
                    "start_s": float(item["start_s"]),
                    "end_s": float(item["end_s"]),
                    "prob_cough": float(p[cough_idx]),
                    "hfc": float(item["hfc"]),
                    "flux": float(item["flux"]),
                    "band_ratio": float(item["band_ratio"]),
                    "flatness": float(item["flatness"]),
                }
            )
        feats = []
        meta = []

    for i, s in enumerate(tqdm(starts, desc="[scan_hfc]")):
        seg = y[s:s + seg_len]
        seg = fix_length(seg, seg_len)

        feat = compute_logmel_cfg(seg, sr=sr, logmel_cfg=logmel_cfg)
        hfc, flux, band_ratio, flatness = compute_aux_metrics(
            seg,
            sr=sr,
            logmel_cfg=logmel_cfg,
            band_lo=float(args.band_lo),
            band_hi=float(args.band_hi),
        )

        feats.append(feat)
        meta.append(
            {
                "frame_idx": i,
                "start_s": s / sr,
                "end_s": (s + seg_len) / sr,
                "hfc": hfc,
                "flux": flux,
                "band_ratio": band_ratio,
                "flatness": flatness,
            }
        )

        if len(feats) >= int(args.batch_size):
            flush_batch()

    flush_batch()

    frame_df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)

    frame_df["hfc_z"] = robust_z(frame_df["hfc"].to_numpy())
    frame_df["flux_z"] = robust_z(frame_df["flux"].to_numpy())
    frame_df["band_ratio_z"] = robust_z(frame_df["band_ratio"].to_numpy())
    frame_df["flatness_z"] = robust_z(frame_df["flatness"].to_numpy())

    frame_df["hfc_s"] = sigmoid01(frame_df["hfc_z"].to_numpy())
    frame_df["flux_s"] = sigmoid01(frame_df["flux_z"].to_numpy())
    frame_df["band_ratio_s"] = sigmoid01(frame_df["band_ratio_z"].to_numpy())
    frame_df["flatness_s"] = sigmoid01(frame_df["flatness_z"].to_numpy())

    aux_score = (
        0.40 * frame_df["hfc_s"].to_numpy()
        + 0.30 * frame_df["flux_s"].to_numpy()
        + 0.20 * frame_df["band_ratio_s"].to_numpy()
        + 0.10 * (1.0 - frame_df["flatness_s"].to_numpy())
    )

    frame_df["aux_score"] = aux_score.astype(np.float32)
    frame_df["score_final"] = (
        0.70 * frame_df["prob_cough"].to_numpy()
        + 0.30 * frame_df["aux_score"].to_numpy()
    ).astype(np.float32)

    frame_df["score_smooth"] = smooth_1d(frame_df["score_final"].to_numpy(), win=int(args.smooth))

    out_frames = f"{args.out_prefix}_frames.csv"
    frame_df.to_csv(out_frames, index=False, encoding="utf-8-sig")
    print("[OK] wrote frames ->", os.path.abspath(out_frames))

    event_df = merge_events_hysteresis(
        frame_df,
        seg_s=seg_s,
        high_thr=float(args.thr),
        low_thr=float(args.thr_low),
        merge_gap_s=float(args.merge_gap),
        min_event_s=float(args.min_event),
        score_col="score_smooth",
    )

    out_events = f"{args.out_prefix}_events.csv"
    event_df.to_csv(out_events, index=False, encoding="utf-8-sig")
    print("[OK] wrote events ->", os.path.abspath(out_events))
    print(
        f"[DONE] frames={len(frame_df)} events={len(event_df)} "
        f"thr={args.thr} thr_low={args.thr_low} smooth={args.smooth}"
    )


if __name__ == "__main__":
    main()