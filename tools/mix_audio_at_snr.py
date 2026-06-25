from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OFFSET_KEY_VERSION = "demand_noise_offset_v2"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def deterministic_seed(parts: Sequence[object]) -> int:
    """Return a process-stable 63-bit seed from SHA256 over the exact key parts."""
    key = "\x1f".join(str(x) for x in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


def noise_offset_key(
    *,
    fold: int,
    model_seed: int | None,
    target_active_snr_db: float | None,
    clean_path: str,
    clean_md5: str,
    environment_recording_id: str,
    noise_sha256: str,
    global_noise_seed: int,
    noise_repeat: int,
) -> dict[str, object]:
    """Build the protocol v2 noise draw key.

    ``model_seed`` and ``target_active_snr_db`` are accepted for audit context,
    but deliberately excluded from the key so the same clean sample and DEMAND
    environment uses the same noise segment across model seeds and SNRs.
    """
    key_parts = [
        f"offset_key_version={OFFSET_KEY_VERSION}",
        f"fold={int(fold)}",
        f"clean_path={str(clean_path)}",
        f"clean_md5={str(clean_md5).strip().lower()}",
        f"environment_recording_id={str(environment_recording_id).strip().upper()}",
        f"noise_sha256={str(noise_sha256).strip().lower()}",
        f"global_noise_seed={int(global_noise_seed)}",
        f"noise_repeat={int(noise_repeat)}",
    ]
    key_text = "\x1f".join(key_parts)
    return {
        "offset_key_version": OFFSET_KEY_VERSION,
        "noise_draw_id": hashlib.sha256(key_text.encode("utf-8")).hexdigest(),
        "noise_repeat": int(noise_repeat),
        "key_parts": key_parts,
        "ignored_model_seed": None if model_seed is None else int(model_seed),
        "ignored_target_active_snr_db": None if target_active_snr_db is None else float(target_active_snr_db),
    }


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def load_audio_with_valid_region(
    path: str | Path,
    *,
    sr: int = 32000,
    dur_s: float = 2.0,
) -> tuple[np.ndarray, int, int, int]:
    """Replicate training audio loading and return the non-padding region.

    The returned waveform has exactly ``sr * dur_s`` samples. For short clips,
    ``valid_start`` and ``valid_samples`` mark the original resampled audio
    inside the centered zero padding. For long clips, the centered crop is fully
    valid.
    """
    p = Path(path)
    y, source_sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {p}")
    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if int(source_sr) != int(sr):
        g = math.gcd(int(source_sr), int(sr))
        y = resample_poly(y, int(sr) // g, int(source_sr) // g).astype(np.float32)
        source_sr = int(sr)
    target_len = int(round(sr * dur_s))
    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
        valid_start = left
        valid_samples = target_len - pad
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]
        valid_start = 0
        valid_samples = target_len
    else:
        valid_start = 0
        valid_samples = target_len
    return y.astype(np.float32), int(source_sr), int(valid_start), int(valid_samples)


def load_noise_channel(
    path: str | Path,
    *,
    selected_channel: int = 1,
    target_sr: int = 32000,
) -> tuple[np.ndarray, int, int]:
    p = Path(path)
    y, source_sr = sf.read(str(p), dtype="float32", always_2d=True)
    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length noise audio: {p}")
    if selected_channel < 1 or selected_channel > y.shape[1]:
        raise ValueError(f"selected_channel={selected_channel} outside 1..{y.shape[1]} for {p}")
    out = np.asarray(y[:, selected_channel - 1], dtype=np.float32)
    if int(source_sr) != int(target_sr):
        g = math.gcd(int(source_sr), int(target_sr))
        out = resample_poly(out, int(target_sr) // g, int(source_sr) // g).astype(np.float32)
    return out.astype(np.float32), int(source_sr), int(target_sr)


def select_noise_segment(
    noise: np.ndarray,
    target_samples: int,
    key_parts: Sequence[object],
    *,
    return_record: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    arr = np.asarray(noise, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot select segment from empty noise waveform.")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")
    if arr.size >= target_samples:
        max_offset = int(arr.size - target_samples)
        seed = deterministic_seed(key_parts)
        offset = int(seed % (max_offset + 1))
        segment = arr[offset : offset + target_samples]
    else:
        seed = deterministic_seed(key_parts)
        offset = 0
        reps = int(math.ceil(target_samples / arr.size))
        segment = np.tile(arr, reps)[:target_samples]
    record = {"noise_seed": int(seed), "noise_offset": int(offset)}
    return (segment.astype(np.float32), record) if return_record else segment.astype(np.float32)


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def mix_clean_with_noise_at_snr(
    clean: np.ndarray,
    noise_segment: np.ndarray,
    *,
    valid_start: int,
    valid_samples: int,
    target_snr_db: float,
    eps: float = 1e-8,
    peak_limit: float = 0.999,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise_segment, dtype=np.float32)
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {clean.shape} vs {noise.shape}")
    if valid_samples <= 0:
        raise ValueError("valid_samples must be positive.")
    stop = int(valid_start) + int(valid_samples)
    if valid_start < 0 or stop > clean.size:
        raise ValueError("valid_start/valid_samples outside waveform bounds.")
    active = slice(int(valid_start), stop)
    clean_active_rms = _rms(clean[active])
    noise_active_rms = _rms(noise[active])
    if clean_active_rms <= eps:
        raise ValueError(f"clean_active_rms={clean_active_rms} <= eps={eps}")
    if noise_active_rms <= eps:
        raise ValueError(f"noise_active_rms_before_gain={noise_active_rms} <= eps={eps}")
    target_noise_rms = clean_active_rms / (10.0 ** (float(target_snr_db) / 20.0))
    gain = target_noise_rms / noise_active_rms
    scaled_noise = noise * np.float32(gain)
    mixed_before_scale = clean + scaled_noise
    peak_before_scale = float(np.max(np.abs(mixed_before_scale)))
    clipping_detected = peak_before_scale > peak_limit
    final_global_scale = min(1.0, peak_limit / peak_before_scale) if peak_before_scale > 0 else 1.0
    mixed = mixed_before_scale * np.float32(final_global_scale)
    clean_scaled = clean * np.float32(final_global_scale)
    noise_scaled = scaled_noise * np.float32(final_global_scale)
    achieved_active = 20.0 * math.log10(_rms(clean_scaled[active]) / max(_rms(noise_scaled[active]), eps))
    achieved_full = 20.0 * math.log10(_rms(clean_scaled) / max(_rms(noise_scaled), eps))
    return mixed.astype(np.float32), {
        "original_valid_samples": int(valid_samples),
        "valid_start": int(valid_start),
        "valid_duration_ratio": float(valid_samples / clean.size),
        "snr_reference": "active_valid_region",
        "target_active_snr_db": float(target_snr_db),
        "target_snr_db": float(target_snr_db),
        "clean_active_rms": float(clean_active_rms),
        "noise_active_rms_before_gain": float(noise_active_rms),
        "gain": float(gain),
        "achieved_active_snr_db": float(achieved_active),
        "achieved_full_window_snr_db": float(achieved_full),
        "peak_before_scale": float(peak_before_scale),
        "final_global_scale": float(final_global_scale),
        "clipping_detected": bool(clipping_detected),
    }


def demand_environment_recording_id(environment: str, channel_file: str) -> str:
    return f"DEMAND:{str(environment).strip().upper()}"


def validate_noise_source_for_paper(row: Mapping[str, object], *, paper_facing: bool) -> bool:
    if not paper_facing:
        return True
    dataset = str(row.get("dataset", "")).strip().lower()
    allowed_publication = _as_bool(row.get("allowed_for_publication", False))
    allowed_redistribution = _as_bool(row.get("allowed_for_redistribution", False))
    if dataset == "demand" and allowed_publication and allowed_redistribution:
        return True
    raise ValueError(
        "Noise source is not paper-facing: "
        f"dataset={row.get('dataset')}, allowed_for_publication={row.get('allowed_for_publication')}, "
        f"allowed_for_redistribution={row.get('allowed_for_redistribution')}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mix one clean audio file with one noise file at a deterministic SNR.")
    ap.add_argument("--clean_audio", required=True)
    ap.add_argument("--noise_audio", required=True)
    ap.add_argument("--snr_db", type=float, required=True)
    ap.add_argument("--out_wav", default="", help="Optional output WAV. By default no mixed audio is written.")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--dur_s", type=float, default=2.0)
    ap.add_argument("--selected_channel", type=int, default=1)
    ap.add_argument("--key", nargs="*", default=[])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    clean, sr, valid_start, valid_samples = load_audio_with_valid_region(args.clean_audio, sr=args.sr, dur_s=args.dur_s)
    noise, source_sr, _ = load_noise_channel(args.noise_audio, selected_channel=args.selected_channel, target_sr=args.sr)
    segment, offset_record = select_noise_segment(
        noise,
        clean.size,
        [args.clean_audio, args.noise_audio, *args.key],
        return_record=True,
    )
    mixed, mix_record = mix_clean_with_noise_at_snr(
        clean,
        segment,
        valid_start=valid_start,
        valid_samples=valid_samples,
        target_snr_db=args.snr_db,
    )
    record = {
        **offset_record,
        **mix_record,
        "clean_audio": str(Path(args.clean_audio)),
        "noise_audio": str(Path(args.noise_audio)),
        "selected_channel": int(args.selected_channel),
        "clean_sr": int(sr),
        "noise_original_sr": int(source_sr),
        "mixed_written": bool(args.out_wav),
    }
    if args.out_wav:
        sf.write(args.out_wav, mixed, args.sr)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
