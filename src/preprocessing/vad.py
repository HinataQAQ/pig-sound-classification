from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks


def _fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant").astype(np.float32)


def fixed_length_crop(y: np.ndarray, target_len: int, train: bool = False) -> Tuple[np.ndarray, int]:
    if len(y) <= target_len:
        return _fix_length(y, target_len), 0
    if train:
        start = int(np.random.randint(0, len(y) - target_len + 1))
    else:
        start = max(0, (len(y) - target_len) // 2)
    return y[start:start + target_len].astype(np.float32), int(start)


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def compute_activity_score(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 1024,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    band_fmin: float = 600.0,
    band_fmax: float = 3000.0,
) -> Dict[str, np.ndarray]:
    if y.size == 0:
        y = np.zeros(1, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    hop_length = max(1, int(round(sr * hop_ms / 1000.0)))
    win_length = max(1, int(round(sr * frame_ms / 1000.0)))
    if n_fft < win_length:
        n_fft = 1 << (win_length - 1).bit_length()

    rms = librosa.feature.rms(
        y=y,
        frame_length=win_length,
        hop_length=hop_length,
        center=False,
    )[0]

    fmax_eff = min(float(fmax), sr / 2.0)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=int(n_mels),
        fmin=float(fmin),
        fmax=fmax_eff,
        power=2.0,
        center=False,
    )
    mel_freqs = librosa.mel_frequencies(n_mels=int(n_mels), fmin=float(fmin), fmax=fmax_eff)
    band_idx = np.where((mel_freqs >= float(band_fmin)) & (mel_freqs <= min(float(band_fmax), fmax_eff)))[0]
    if band_idx.size == 0:
        band_idx = np.arange(mel.shape[0])

    band_energy = np.log(mel[band_idx].mean(axis=0) + 1e-12)
    flux = np.maximum(np.diff(band_energy, prepend=band_energy[0]), 0.0)

    T = int(min(len(rms), len(band_energy), len(flux)))
    rms = rms[:T]
    band_energy = band_energy[:T]
    flux = flux[:T]

    score = _robust_z(np.log(rms + 1e-8)) + 0.75 * _robust_z(band_energy) + 0.50 * _robust_z(flux)
    times_s = (np.arange(T, dtype=np.float32) * hop_length + 0.5 * win_length) / float(sr)

    return {
        "times_s": times_s.astype(np.float32),
        "score": score.astype(np.float32),
        "rms": rms.astype(np.float32),
        "band_energy": band_energy.astype(np.float32),
        "flux": flux.astype(np.float32),
        "hop_length": int(hop_length),
        "win_length": int(win_length),
        "n_fft": int(n_fft),
    }


def detect_activity_regions(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 1024,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    band_fmin: float = 600.0,
    band_fmax: float = 3000.0,
    score_threshold: float = 1.2,
    min_event_ms: float = 80.0,
    merge_gap_ms: float = 120.0,
    prepad_ms: float = 80.0,
    postpad_ms: float = 150.0,
) -> Dict[str, object]:
    info = compute_activity_score(
        y=y,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        band_fmin=band_fmin,
        band_fmax=band_fmax,
    )
    score = info["score"]
    hop_length = int(info["hop_length"])
    win_length = int(info["win_length"])

    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med)) + 1e-6)
    thr = med + float(score_threshold) * 1.4826 * mad

    active = score >= thr
    peak_distance = max(1, int(round(float(min_event_ms) / float(hop_ms))))
    peaks, peak_props = find_peaks(score, height=thr, distance=peak_distance)
    for p in peaks:
        lo = max(0, int(p) - 1)
        hi = min(active.shape[0], int(p) + 2)
        active[lo:hi] = True

    gap_frames = max(1, int(round(float(merge_gap_ms) / float(hop_ms))))
    min_frames = max(1, int(round(float(min_event_ms) / float(hop_ms))))

    segments: List[List[int]] = []
    start = None
    for i, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            segments.append([start, i - 1])
            start = None
    if start is not None:
        segments.append([start, len(active) - 1])

    merged: List[List[int]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        if seg[0] - merged[-1][1] <= gap_frames:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    prepad = int(round(float(prepad_ms) * sr / 1000.0))
    postpad = int(round(float(postpad_ms) * sr / 1000.0))

    regions = []
    for s_idx, e_idx in merged:
        if (e_idx - s_idx + 1) < min_frames:
            continue
        local = score[s_idx:e_idx + 1]
        peak_rel = int(np.argmax(local))
        peak_idx = s_idx + peak_rel
        start_sample = max(0, int(s_idx * hop_length) - prepad)
        end_sample = min(len(y), int(e_idx * hop_length + win_length) + postpad)
        regions.append(
            {
                "start": int(start_sample),
                "end": int(end_sample),
                "start_s": float(start_sample / sr),
                "end_s": float(end_sample / sr),
                "peak_idx": int(peak_idx),
                "peak_time_s": float(info["times_s"][peak_idx]),
                "peak_score": float(score[peak_idx]),
                "duration_s": float((end_sample - start_sample) / float(sr)),
            }
        )

    info["threshold"] = float(thr)
    info["active"] = active.astype(np.bool_)
    info["peak_indices"] = peaks.astype(np.int32)
    info["peak_scores"] = peak_props.get("peak_heights", np.array([], dtype=np.float32)).astype(np.float32)
    info["regions"] = regions
    return info


def crop_around_activity(
    y: np.ndarray,
    sr: int,
    target_len: int,
    train: bool = False,
    **vad_kwargs,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(y) <= target_len:
        return _fix_length(y, target_len), {"used_vad": False, "regions": []}

    info = detect_activity_regions(y=y, sr=sr, **vad_kwargs)
    regions = info.get("regions", [])

    if not regions:
        crop, start = fixed_length_crop(y, target_len, train=train)
        return crop, {
            "used_vad": False,
            "regions": [],
            "crop_start": int(start),
            "crop_end": int(start + target_len),
        }

    if train and len(regions) > 1:
        weights = np.array([max(r["peak_score"], 1e-3) for r in regions], dtype=np.float64)
        weights /= weights.sum()
        chosen = regions[int(np.random.choice(len(regions), p=weights))]
    else:
        chosen = max(regions, key=lambda r: (r["peak_score"], r["duration_s"]))

    center = 0.5 * (float(chosen["start"]) + float(chosen["end"]))
    if train:
        jitter = int(round(0.10 * target_len))
        center += int(np.random.randint(-jitter, jitter + 1))

    start = int(round(center - target_len / 2.0))
    start = min(max(0, start), max(0, len(y) - target_len))
    crop = _fix_length(y[start:start + target_len], target_len)

    return crop, {
        "used_vad": True,
        "regions": regions,
        "chosen_region": chosen,
        "crop_start": int(start),
        "crop_end": int(start + target_len),
    }
