from __future__ import annotations

import json
import math
import posixpath
import re
import hashlib
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAIN_LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]
DEFAULT_AUX_LABELS = [
    "dry_cough",
    "abdominal_cough",
    "calm_grunt",
    "feeding",
    "frightened_stress",
    "anxious_stress",
]
DEFAULT_AUX_TO_MAIN = {
    "dry_cough": "cough",
    "abdominal_cough": "cough",
    "calm_grunt": "calm_grunt",
    "feeding": "feeding",
    "frightened_stress": "stress_vocal",
    "anxious_stress": "stress_vocal",
}


@dataclass(frozen=True)
class HierModelConfig:
    main_labels: list[str]
    aux_labels: list[str]
    feature_mode: str = "logmel"
    use_se: bool = True
    seed: int | None = None
    sr: int = 32000
    dur_s: float = 2.0
    n_mels: int = 64
    n_mfcc: int = 20
    fmin: float = 50.0
    fmax: float = 8000.0
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 800
    rnn_type: str = "gru"
    pooling_type: str = "mean"
    hier_aux_weight: float | None = None


def require_file(path: str | Path, description: str = "file") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {description}: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Expected {description} to be a file: {p}")
    return p


def project_relative_path(path: str | Path, root: str | Path = ROOT) -> str | None:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        return None


def path_record(path: str | Path, root: str | Path = ROOT) -> dict[str, str | None]:
    p = Path(path)
    return {
        "project_relative": project_relative_path(p, root=root),
        "original_absolute": str(p.resolve()),
        "sha256": file_sha256(p) if p.exists() and p.is_file() else None,
    }


def resolve_recorded_file(
    record: Mapping[str, Any],
    *,
    absolute_key: str,
    relative_key: str | None = None,
    sha256_key: str | None = None,
    description: str = "recorded file",
    root: str | Path = ROOT,
) -> Path:
    candidates: list[Path] = []
    project_paths = record.get("project_relative_paths")
    if relative_key and isinstance(project_paths, Mapping):
        rel = project_paths.get(relative_key)
        if rel:
            candidates.append(Path(root) / str(rel))
    paths_record = record.get("paths")
    if relative_key and isinstance(paths_record, Mapping):
        payload = paths_record.get(relative_key)
        if isinstance(payload, Mapping) and payload.get("project_relative"):
            candidates.append(Path(root) / str(payload["project_relative"]))
    absolute = record.get(absolute_key)
    if absolute:
        candidates.append(Path(str(absolute)))

    expected_sha = str(record.get(sha256_key, "")) if sha256_key else ""
    if not expected_sha and relative_key and isinstance(paths_record, Mapping):
        payload = paths_record.get(relative_key)
        if isinstance(payload, Mapping) and payload.get("sha256"):
            expected_sha = str(payload["sha256"])

    seen: set[str] = set()
    existing: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            existing.append(candidate)

    if not existing:
        raise FileNotFoundError(
            f"Missing {description}; tried: {[str(x) for x in candidates]}"
        )
    if expected_sha:
        for candidate in existing:
            if file_sha256(candidate) == expected_sha:
                return candidate
        raise RuntimeError(f"{description} exists but SHA256 does not match recorded metadata.")
    return existing[0]


def normalize_identity_path(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    text = posixpath.normpath(text)
    if text == ".":
        text = ""
    elif text.startswith("./"):
        text = text[2:]
    return text.lower()


def recorded_manifest_sha(metadata: Mapping[str, Any], split: str) -> str | None:
    key = f"{split}_manifest"
    manifest_sha = metadata.get("manifest_sha256")
    if isinstance(manifest_sha, Mapping) and manifest_sha.get(split):
        return str(manifest_sha[split])

    paths_record = metadata.get("paths")
    if isinstance(paths_record, Mapping):
        payload = paths_record.get(key)
        if isinstance(payload, Mapping) and payload.get("sha256"):
            return str(payload["sha256"])

    input_hashes = metadata.get("input_hashes")
    if isinstance(input_hashes, Mapping):
        payload = input_hashes.get(key)
        if isinstance(payload, Mapping) and payload.get("sha256"):
            return str(payload["sha256"])

    direct_key = f"{key}_sha256"
    if metadata.get(direct_key):
        return str(metadata[direct_key])
    return None


def verify_file_sha256(path: str | Path, expected_sha256: str | None, description: str) -> str:
    actual = file_sha256(path)
    if expected_sha256 and actual.lower() != str(expected_sha256).lower():
        raise RuntimeError(
            f"{description} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def verify_manifest_matches_metadata(
    manifest_path: str | Path,
    metadata: Mapping[str, Any],
    split: str,
) -> str:
    expected = recorded_manifest_sha(metadata, split)
    if not expected:
        raise RuntimeError(f"Missing recorded {split} manifest SHA256 in metadata.")
    return verify_file_sha256(manifest_path, expected, f"{split} manifest")


def canonical_feature_backend(feature_backend: str) -> str:
    backend = str(feature_backend).strip()
    if backend == "librosa":
        return "training_exact"
    return backend


def result_qualification_fields(
    feature_backend: str,
    *,
    fold: int | None,
    seed: int | None,
    input_role: str | None = None,
    unsafe_allow_checkpoint_sha_mismatch: bool = False,
    manifest_sha_verified: bool = True,
    leakage_audit_ok: bool = True,
    run_scope: str = "fold_seed",
) -> dict[str, Any]:
    backend = canonical_feature_backend(feature_backend)
    feature_pipeline_equivalent = backend == "training_exact"
    single_fold_debug = fold is not None and seed is not None and int(fold) == 0 and int(seed) == 3407
    frozen_test_ok = input_role is None or str(input_role) == "frozen_test"
    eligible = (
        feature_pipeline_equivalent
        and frozen_test_ok
        and not bool(unsafe_allow_checkpoint_sha_mismatch)
        and bool(manifest_sha_verified)
        and bool(leakage_audit_ok)
    )
    return {
        "run_scope": run_scope,
        "feature_pipeline_equivalent": bool(feature_pipeline_equivalent),
        "single_fold_debug": bool(single_fold_debug),
        "eligible_for_cv_aggregation": bool(eligible),
        "paper_main_result": False,
        "feature_backend": backend,
    }


def select_prediction_input_role(
    *,
    test_manifest: str | Path | None,
    manifest: str | Path | None,
    audio: str | Path | None,
) -> dict[str, str]:
    provided = [
        ("frozen_test", test_manifest),
        ("inference_manifest", manifest),
        ("single_audio", audio),
    ]
    selected = [(role, value) for role, value in provided if value is not None and str(value) != ""]
    if len(selected) != 1:
        raise ValueError("Provide exactly one of --test_manifest, --manifest, or --audio.")
    role, value = selected[0]
    return {"input_role": role, "input_path": str(value)}


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def calibration_float_param(
    calibration: Mapping[str, Any],
    method: str,
    param: str,
    *,
    fallback_keys: Sequence[str] = (),
    default: float | None = None,
) -> float:
    method_params = calibration.get("method_best_params", {})
    if isinstance(method_params, Mapping):
        payload = method_params.get(method)
        if isinstance(payload, Mapping):
            value = payload.get(param)
            if not _value_is_missing(value):
                return float(value)
    for key in fallback_keys:
        value = calibration.get(key)
        if not _value_is_missing(value):
            return float(value)
    if default is None:
        raise RuntimeError(f"Missing calibration parameter {method}.{param}")
    return float(default)


def normalize_source_id(value: Any) -> str:
    return str(value).strip()


def normalize_md5(value: Any) -> str:
    return str(value).strip().lower()


def require_dir(path: str | Path, description: str = "directory", must_be_new: bool = False) -> Path:
    p = Path(path)
    if must_be_new and p.exists():
        raise FileExistsError(f"Refusing to overwrite existing {description}: {p}")
    p.mkdir(parents=True, exist_ok=not must_be_new)
    return p


def read_json(path: str | Path) -> dict[str, Any]:
    p = require_file(path, "JSON file")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = require_file(path, "file to hash")
    digest = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256_map(paths: Mapping[str, str | Path | None]) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for key, value in paths.items():
        if value is None or str(value) == "":
            out[str(key)] = {"path": None, "sha256": None}
            continue
        p = require_file(value, key)
        out[str(key)] = {"path": str(p.resolve()), "sha256": file_sha256(p)}
    return out


def parse_csv_floats(text: str, *, name: str) -> list[float]:
    vals: list[float] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        vals.append(float(raw))
    if not vals:
        raise ValueError(f"{name} must contain at least one numeric value.")
    return vals


def infer_fold_seed(*values: str | Path) -> dict[str, int | None]:
    joined = " ".join(str(v) for v in values if v is not None)
    fold_match = re.search(r"fold(\d+)", joined, flags=re.IGNORECASE)
    seed_match = re.search(r"seed(\d+)", joined, flags=re.IGNORECASE)
    return {
        "fold": int(fold_match.group(1)) if fold_match else None,
        "seed": int(seed_match.group(1)) if seed_match else None,
    }


def config_from_summary(summary: Mapping[str, Any]) -> HierModelConfig:
    main_labels = [str(x).strip().lower() for x in summary.get("main_labels", DEFAULT_MAIN_LABELS)]
    aux_labels = [str(x).strip().lower() for x in summary.get("aux_labels", DEFAULT_AUX_LABELS)]
    return HierModelConfig(
        main_labels=main_labels,
        aux_labels=aux_labels,
        feature_mode=str(summary.get("feature_mode", "logmel")),
        use_se=bool(summary.get("use_se", True)),
        seed=int(summary["seed"]) if summary.get("seed") is not None else None,
        sr=int(summary.get("sr", 32000)),
        dur_s=float(summary.get("dur_s", 2.0)),
        n_mels=int(summary.get("n_mels", 64)),
        n_mfcc=int(summary.get("n_mfcc", 20)),
        fmin=float(summary.get("fmin", 50.0)),
        fmax=float(summary.get("fmax", 8000.0)),
        n_fft=int(summary.get("n_fft", 1024)),
        hop_length=int(summary.get("hop_length", 320)),
        win_length=int(summary.get("win_length", 800)),
        rnn_type=str(summary.get("rnn_type", "gru")),
        pooling_type=str(summary.get("pooling_type", "mean")),
        hier_aux_weight=(
            float(summary["hier_aux_weight"])
            if summary.get("hier_aux_weight") is not None
            else None
        ),
    )


def load_config(summary_path: str | Path | None = None, config_path: str | Path | None = None) -> HierModelConfig:
    if summary_path is None and config_path is None:
        raise ValueError("Either summary_path or config_path is required to recover model configuration.")

    payload = read_json(summary_path if summary_path is not None else config_path)
    if "model_config" in payload and isinstance(payload["model_config"], dict):
        payload = payload["model_config"]
    return config_from_summary(payload)


def config_to_jsonable(config: HierModelConfig) -> dict[str, Any]:
    return asdict(config)


def training_exact_feature_config(config: HierModelConfig) -> dict[str, Any]:
    return {
        "feature_backend": "training_exact",
        "feature_mode": config.feature_mode,
        "sr": int(config.sr),
        "dur_s": float(config.dur_s),
        "n_mels": int(config.n_mels),
        "n_mfcc": int(config.n_mfcc),
        "n_fft": int(config.n_fft),
        "hop_length": int(config.hop_length),
        "win_length": int(config.win_length),
        "fmin": float(config.fmin),
        "fmax": float(config.fmax),
        "window": "hann",
        "center": True,
        "pad_mode": "constant",
        "power": 2.0,
        "mel_htk": False,
        "mel_norm": "slaney",
        "power_to_db_ref": 1.0,
        "power_to_db_top_db": 80.0,
        "waveform_mono_conversion": "soundfile always_2d=False; if ndarray ndim==2, average channels",
        "resampling": "scipy.signal.resample_poly when source_sr != target sr",
        "center_crop": "middle crop to int(sr * dur_s)",
        "zero_padding": "symmetric constant zero padding to int(sr * dur_s)",
        "norm_map": "per-sample z-score with eps=1e-6",
        "source_functions": [
            "train_mctafd_crnn_ablation.load_audio_soundfile",
            "train_mctafd_crnn_ablation.make_feature",
            "train_mctafd_crnn_ablation.norm_map",
            "train_hier_longcontext_crnn.HierPigVocalDataset",
        ],
    }


def feature_config_hash(feature_config: Mapping[str, Any]) -> str:
    payload = json.dumps(feature_config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fusion_formula_metadata() -> dict[str, str]:
    return {
        "fusion_formula": "fused = alpha * softmax + (1 - alpha) * prototype",
        "fusion_weight_name": "alpha / softmax_weight",
        "fusion_weight_alpha_0": "pure prototype",
        "fusion_weight_alpha_1": "pure softmax",
        "hierarchy_penalty_formula": (
            "penalized_confidence = raw_confidence if hierarchy_inconsistent is false; "
            "raw_confidence * hierarchy_confidence_penalty otherwise"
        ),
        "hierarchy_inconsistent_definition": (
            "main prototype prediction differs from map(auxiliary prototype prediction)"
        ),
    }


def validate_expected_config(
    config: HierModelConfig,
    *,
    fold: int | None = None,
    seed: int | None = None,
    expected_hier_aux_weight: float | None = None,
    expected_dur_s: float | None = None,
    expected_feature_mode: str | None = None,
    expected_main_labels: Sequence[str] | None = None,
    expected_aux_labels: Sequence[str] | None = None,
    atol: float = 1e-6,
) -> None:
    errors: list[str] = []

    if seed is not None and config.seed is not None and int(seed) != int(config.seed):
        errors.append(f"seed mismatch: expected {seed}, summary has {config.seed}")
    if expected_hier_aux_weight is not None:
        if config.hier_aux_weight is None:
            errors.append("hier_aux_weight missing from model configuration")
        elif not math.isclose(float(config.hier_aux_weight), float(expected_hier_aux_weight), abs_tol=atol):
            errors.append(
                f"hier_aux_weight mismatch: expected {expected_hier_aux_weight}, summary has {config.hier_aux_weight}"
            )
    if expected_dur_s is not None and not math.isclose(float(config.dur_s), float(expected_dur_s), abs_tol=atol):
        errors.append(f"dur_s mismatch: expected {expected_dur_s}, summary has {config.dur_s}")
    if expected_feature_mode is not None:
        got = str(config.feature_mode).strip().lower()
        expected = str(expected_feature_mode).strip().lower()
        if got != expected:
            errors.append(f"feature_mode mismatch: expected {expected}, summary has {got}")
    if expected_main_labels is not None:
        expected = [str(x).strip().lower() for x in expected_main_labels]
        if list(config.main_labels) != expected:
            errors.append(f"main label order mismatch: expected {expected}, summary has {config.main_labels}")
    if expected_aux_labels is not None:
        expected = [str(x).strip().lower() for x in expected_aux_labels]
        if list(config.aux_labels) != expected:
            errors.append(f"auxiliary label order mismatch: expected {expected}, summary has {config.aux_labels}")

    if fold is not None and int(fold) < 0:
        errors.append(f"fold must be non-negative, got {fold}")

    if errors:
        raise ValueError("Invalid prototype model configuration: " + "; ".join(errors))


def validate_fold_seed_sources(
    *,
    fold: int | None,
    seed: int | None,
    sources: Mapping[str, str | Path | None],
) -> dict[str, dict[str, int | None]]:
    report: dict[str, dict[str, int | None]] = {}
    errors: list[str] = []
    for name, value in sources.items():
        if value is None or str(value) == "":
            continue
        inferred = infer_fold_seed(value)
        report[str(name)] = inferred
        if fold is not None and inferred["fold"] is not None and int(inferred["fold"]) != int(fold):
            errors.append(f"{name} fold mismatch: expected fold{fold}, path suggests fold{inferred['fold']}")
        if seed is not None and inferred["seed"] is not None and int(inferred["seed"]) != int(seed):
            errors.append(f"{name} seed mismatch: expected seed{seed}, path suggests seed{inferred['seed']}")
    if errors:
        raise ValueError("Fold/seed source mismatch: " + "; ".join(errors))
    return report


def instantiate_hier_model(config: HierModelConfig, device: str):
    from train_hier_longcontext_crnn import HierCRNN, feature_channels

    model = HierCRNN(
        num_main_classes=len(config.main_labels),
        num_aux_classes=len(config.aux_labels),
        in_channels=feature_channels(config.feature_mode),
        use_se=config.use_se,
        rnn_type=config.rnn_type,
        pooling_type=config.pooling_type,
    )
    return model.to(device)


def extract_state_dict(raw_checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(raw_checkpoint, dict):
        if "model" in raw_checkpoint and isinstance(raw_checkpoint["model"], dict):
            return raw_checkpoint["model"]
        if "state_dict" in raw_checkpoint and isinstance(raw_checkpoint["state_dict"], dict):
            return raw_checkpoint["state_dict"]
    return raw_checkpoint


def load_hier_model(ckpt_path: str | Path, config: HierModelConfig, device: str = "cpu"):
    import torch

    ckpt = require_file(ckpt_path, "hierarchical checkpoint")
    model = instantiate_hier_model(config, device=device)
    raw = torch.load(ckpt, map_location=device)
    state = extract_state_dict(raw)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_hier_dataset(
    manifest_path: str | Path,
    config: HierModelConfig,
    feature_backend: str = "librosa",
):
    if feature_backend == "numpy_logmel":
        return NumpyLogmelHierDataset(manifest_path, config)
    if feature_backend == "training_exact":
        feature_backend = "librosa"
    if feature_backend != "librosa":
        raise ValueError(f"Unknown feature_backend={feature_backend}")

    from train_hier_longcontext_crnn import HierPigVocalDataset

    manifest = require_file(manifest_path, "manifest")
    return HierPigVocalDataset(
        manifest,
        main_labels=config.main_labels,
        aux_labels=config.aux_labels,
        feature_mode=config.feature_mode,
        sr=config.sr,
        dur_s=config.dur_s,
        cache=True,
        n_mels=config.n_mels,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )


def _normalize_map(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mu) / (sd + eps)).astype(np.float32)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / float(right - center)
    return fb


@lru_cache(maxsize=16)
def _centered_hann_window(n_fft: int, win_length: int) -> np.ndarray:
    from scipy.signal.windows import hann

    window = np.zeros(n_fft, dtype=np.float32)
    core = hann(win_length, sym=False).astype(np.float32)
    start = (n_fft - win_length) // 2
    window[start : start + win_length] = core
    return window


def make_numpy_logmel_feature(
    y: np.ndarray,
    *,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    pad = n_fft // 2
    y_pad = np.pad(y, (pad, pad), mode="constant")
    n_frames = 1 + max(0, (len(y_pad) - n_fft) // hop_length)
    if n_frames <= 0:
        raise RuntimeError("Cannot compute Log-Mel feature: audio is too short after padding.")

    stride = y_pad.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        y_pad,
        shape=(n_frames, n_fft),
        strides=(hop_length * stride, stride),
        writeable=False,
    )
    window = _centered_hann_window(n_fft, win_length)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax) @ power.T
    mel = np.maximum(mel, 1e-10)
    logmel = 10.0 * np.log10(mel)
    logmel = np.maximum(logmel, float(logmel.max()) - 80.0)
    return np.stack([_normalize_map(logmel)], axis=0).astype(np.float32)


def load_audio_numpy(path: str | Path, sr: int, dur_s: float) -> tuple[np.ndarray, int]:
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    p = Path(path)
    y, source_sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {p}")
    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)

    if int(source_sr) != int(sr):
        g = gcd(int(source_sr), int(sr))
        y = resample_poly(y, int(sr) // g, int(source_sr) // g).astype(np.float32)

    target_len = int(sr * dur_s)
    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]

    return y.astype(np.float32), int(sr)


class NumpyLogmelHierDataset:
    def __init__(self, manifest_path: str | Path, config: HierModelConfig):
        import torch
        from train_hier_longcontext_crnn import (
            infer_label_col,
            infer_path_col,
            infer_subtype_from_row,
            normalize_label,
            resolve_path,
        )

        if config.feature_mode != "logmel":
            raise ValueError("feature_backend=numpy_logmel only supports feature_mode=logmel.")

        self._torch = torch
        self._normalize_label = normalize_label
        self._resolve_path = resolve_path
        self.df = pd.read_csv(require_file(manifest_path, "manifest"))
        self.path_col = infer_path_col(self.df)
        self.label_col = infer_label_col(self.df)
        self.main_labels = [x.strip().lower() for x in config.main_labels]
        self.aux_labels = [x.strip().lower() for x in config.aux_labels]
        self.main2id = {x: i for i, x in enumerate(self.main_labels)}
        self.aux2id = {x: i for i, x in enumerate(self.aux_labels)}
        self.config = config
        self.cache = True
        self._cache: dict[int, tuple[Any, Any, Any]] = {}

        self.df[self.label_col] = self.df[self.label_col].astype(str).str.strip().str.lower()
        self.df = self.df[self.df[self.label_col].isin(self.main2id.keys())].reset_index(drop=True)

        aux_list: list[str] = []
        keep_aux: list[bool] = []
        for _, row in self.df.iterrows():
            subtype = infer_subtype_from_row(row, self.path_col, self.label_col)
            aux_list.append(subtype)
            keep_aux.append(subtype in self.aux2id)
        self.df["__aux_label"] = aux_list
        self.df = self.df[keep_aux].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No valid rows after subtype filtering: {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        path = self._resolve_path(row[self.path_col])
        main_lab = self._normalize_label(row[self.label_col])
        aux_lab = self._normalize_label(row["__aux_label"])
        y, sr = load_audio_numpy(path, sr=self.config.sr, dur_s=self.config.dur_s)
        x = make_numpy_logmel_feature(
            y,
            sr=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        item = (
            self._torch.from_numpy(x).float(),
            self._torch.tensor(self.main2id[main_lab], dtype=self._torch.long),
            self._torch.tensor(self.aux2id[aux_lab], dtype=self._torch.long),
        )
        self._cache[idx] = item
        return item


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (arr / denom).astype(np.float32)


def compute_class_prototypes(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    emb = normalize_rows(embeddings)
    label_ids = np.asarray(label_ids, dtype=np.int64)

    prototypes = np.zeros((num_classes, emb.shape[1]), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)

    for class_id in range(num_classes):
        mask = label_ids == class_id
        counts[class_id] = int(mask.sum())
        if counts[class_id] == 0:
            continue
        prototypes[class_id] = emb[mask].mean(axis=0)

    missing = [int(i) for i, count in enumerate(counts) if int(count) == 0]
    if missing:
        raise ValueError(f"Cannot compute prototypes with missing classes: {missing}")

    prototypes = normalize_rows(prototypes)
    return prototypes.astype(np.float32), counts


def cosine_similarities(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    emb = normalize_rows(embeddings)
    proto = normalize_rows(prototypes)
    return emb @ proto.T


def cosine_distances(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return (1.0 - cosine_similarities(embeddings, prototypes)).astype(np.float32)


def euclidean_distances(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    emb = normalize_rows(embeddings)
    proto = normalize_rows(prototypes)
    diff = emb[:, None, :] - proto[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def _numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "max": None,
        }
    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "min": float(np.min(vals)),
        "q05": float(np.quantile(vals, 0.05)),
        "q25": float(np.quantile(vals, 0.25)),
        "q50": float(np.quantile(vals, 0.50)),
        "q75": float(np.quantile(vals, 0.75)),
        "q95": float(np.quantile(vals, 0.95)),
        "max": float(np.max(vals)),
    }


def compute_distance_statistics(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    prototypes: np.ndarray,
    *,
    class_labels: Sequence[str],
    split_name: str,
) -> dict[str, Any]:
    label_ids = np.asarray(label_ids, dtype=np.int64)
    sims = cosine_similarities(embeddings, prototypes)
    cos_dist = (1.0 - sims).astype(np.float32)
    euc_dist = euclidean_distances(embeddings, prototypes)

    classes: dict[str, Any] = {}
    for class_id, label in enumerate(class_labels):
        mask = label_ids == class_id
        own_cos = cos_dist[mask, class_id] if mask.any() else np.asarray([], dtype=np.float32)
        own_euc = euc_dist[mask, class_id] if mask.any() else np.asarray([], dtype=np.float32)
        own_sim = sims[mask, class_id] if mask.any() else np.asarray([], dtype=np.float32)
        classes[str(label)] = {
            "class_id": int(class_id),
            "count": int(mask.sum()),
            "cosine_similarity": _numeric_summary(own_sim),
            "cosine_distance": _numeric_summary(own_cos),
            "euclidean_distance": _numeric_summary(own_euc),
        }

    return {
        "split": str(split_name),
        "distance_definition": {
            "embedding_normalization": "L2",
            "prototype_definition": "L2-normalized mean of L2-normalized training embeddings",
            "primary_distance": "cosine_distance = 1 - cosine_similarity",
            "euclidean_relation_for_normalized_vectors": "euclidean_distance = sqrt(2 - 2 * cosine_similarity)",
        },
        "classes": classes,
    }


def prototype_sample_frame(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    prototypes: np.ndarray,
    labels: Sequence[str],
    *,
    label_prefix: str,
) -> pd.DataFrame:
    sims = cosine_similarities(embeddings, prototypes)
    cos_dist = (1.0 - sims).astype(np.float32)
    euc_dist = euclidean_distances(embeddings, prototypes)
    label_ids = np.asarray(label_ids, dtype=np.int64)
    nearest = sims.argmax(axis=1)
    nearest_other = np.zeros(len(label_ids), dtype=np.int64)
    nearest_other_sim = np.zeros(len(label_ids), dtype=np.float32)

    for i, class_id in enumerate(label_ids):
        row = sims[i].copy()
        if 0 <= int(class_id) < row.shape[0]:
            row[int(class_id)] = -np.inf
        other_id = int(np.argmax(row))
        nearest_other[i] = other_id
        nearest_other_sim[i] = float(row[other_id])

    own_sim = sims[np.arange(len(label_ids)), label_ids]
    own_cos_dist = cos_dist[np.arange(len(label_ids)), label_ids]
    own_euc_dist = euc_dist[np.arange(len(label_ids)), label_ids]
    frame = metadata.copy().reset_index(drop=True)
    frame[f"{label_prefix}_label_id"] = label_ids
    frame[f"{label_prefix}_label"] = [labels[int(i)] for i in label_ids]
    frame[f"{label_prefix}_nearest_prototype_id"] = nearest
    frame[f"{label_prefix}_nearest_prototype"] = [labels[int(i)] for i in nearest]
    frame[f"{label_prefix}_own_cosine_similarity"] = own_sim
    frame[f"{label_prefix}_own_cosine_distance"] = own_cos_dist
    frame[f"{label_prefix}_own_euclidean_distance"] = own_euc_dist
    frame[f"{label_prefix}_nearest_other_prototype_id"] = nearest_other
    frame[f"{label_prefix}_nearest_other_prototype"] = [labels[int(i)] for i in nearest_other]
    frame[f"{label_prefix}_nearest_other_cosine_similarity"] = nearest_other_sim
    frame[f"{label_prefix}_confusion_margin"] = own_sim - nearest_other_sim
    return frame.copy()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    return (exp / denom).astype(np.float32)


def similarity_to_probabilities(similarities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return stable_softmax(np.asarray(similarities, dtype=np.float32) / float(temperature))


def temperature_scale_probabilities(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    p = normalize_probabilities(probs)
    return stable_softmax(np.log(np.clip(p, 1e-12, 1.0)) / float(temperature))


def normalize_probabilities(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    denom = np.maximum(arr.sum(axis=1, keepdims=True), eps)
    return (arr / denom).astype(np.float32)


def map_aux_label_to_main(aux_label: str) -> str:
    lab = str(aux_label).strip().lower()
    if lab not in DEFAULT_AUX_TO_MAIN:
        raise ValueError(f"Cannot map aux label to main label: {aux_label}")
    return DEFAULT_AUX_TO_MAIN[lab]


def map_aux_probabilities_to_main(
    aux_probs: np.ndarray,
    aux_labels: Sequence[str],
    main_labels: Sequence[str],
) -> np.ndarray:
    main2id = {str(label).strip().lower(): i for i, label in enumerate(main_labels)}
    out = np.zeros((aux_probs.shape[0], len(main_labels)), dtype=np.float32)

    for aux_idx, aux_label in enumerate(aux_labels):
        main_label = map_aux_label_to_main(str(aux_label).strip().lower())
        if main_label not in main2id:
            raise ValueError(f"Mapped main label {main_label!r} is not in main label list.")
        out[:, main2id[main_label]] += aux_probs[:, aux_idx]

    return normalize_probabilities(out)


def hierarchical_main_probabilities(
    main_proto_probs: np.ndarray,
    aux_proto_probs: np.ndarray,
    aux_labels: Sequence[str],
    main_labels: Sequence[str],
    aux_weight: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= aux_weight <= 1.0:
        raise ValueError(f"aux_weight must be in [0, 1], got {aux_weight}")
    mapped_aux = map_aux_probabilities_to_main(aux_proto_probs, aux_labels, main_labels)
    fused = (1.0 - aux_weight) * main_proto_probs + aux_weight * mapped_aux
    return normalize_probabilities(fused)


def hierarchy_inconsistency_flags(
    main_pred: Sequence[str],
    aux_pred: Sequence[str],
) -> np.ndarray:
    mapped = [map_aux_label_to_main(x) for x in aux_pred]
    return np.asarray([str(a) != str(b) for a, b in zip(main_pred, mapped)], dtype=bool)


def apply_hierarchy_confidence_penalty(
    confidence: np.ndarray,
    main_pred: Sequence[str],
    aux_pred: Sequence[str],
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= float(penalty) <= 1.0:
        raise ValueError(f"hierarchy confidence penalty must be in [0, 1], got {penalty}")
    conf = np.asarray(confidence, dtype=np.float32).copy()
    inconsistent = hierarchy_inconsistency_flags(main_pred, aux_pred)
    conf[inconsistent] *= float(penalty)
    return conf.astype(np.float32), inconsistent


def fuse_probabilities(
    softmax_probs: np.ndarray,
    prototype_probs: np.ndarray,
    softmax_weight: float,
) -> np.ndarray:
    if not 0.0 <= softmax_weight <= 1.0:
        raise ValueError(f"softmax_weight must be in [0, 1], got {softmax_weight}")
    return normalize_probabilities(
        softmax_weight * np.asarray(softmax_probs, dtype=np.float32)
        + (1.0 - softmax_weight) * np.asarray(prototype_probs, dtype=np.float32)
    )


def fusion_mode_from_alpha(alpha: float, eps: float = 1e-9) -> str:
    value = float(alpha)
    if abs(value) <= eps:
        return "pure_prototype"
    if abs(value - 1.0) <= eps:
        return "pure_softmax"
    return "mixed"


def calibration_relevant_parameters(
    method: str,
    *,
    prototype_temperature: float | None,
    softmax_temperature: float | None,
    softmax_weight: float | None,
    hier_aux_prob_weight: float | None,
) -> dict[str, Any]:
    method = str(method)
    if method == "raw_softmax":
        return {
            "method": method,
            "prototype_temperature": None,
            "softmax_temperature": None,
            "softmax_weight": None,
            "hier_aux_prob_weight": None,
            "fusion_kind": None,
        }
    if method == "calibrated_softmax":
        return {
            "method": method,
            "prototype_temperature": None,
            "softmax_temperature": float(softmax_temperature) if softmax_temperature is not None else None,
            "softmax_weight": None,
            "hier_aux_prob_weight": None,
            "fusion_kind": None,
        }
    if method == "prototype":
        return {
            "method": method,
            "prototype_temperature": float(prototype_temperature) if prototype_temperature is not None else None,
            "softmax_temperature": None,
            "softmax_weight": None,
            "hier_aux_prob_weight": None,
            "fusion_kind": None,
        }
    if method == "hierarchical":
        return {
            "method": method,
            "prototype_temperature": float(prototype_temperature) if prototype_temperature is not None else None,
            "softmax_temperature": None,
            "softmax_weight": None,
            "hier_aux_prob_weight": float(hier_aux_prob_weight) if hier_aux_prob_weight is not None else None,
            "fusion_kind": None,
        }
    if method == "fused":
        alpha = float(softmax_weight) if softmax_weight is not None else None
        return {
            "method": method,
            "prototype_temperature": float(prototype_temperature) if prototype_temperature is not None else None,
            "softmax_temperature": float(softmax_temperature) if softmax_temperature is not None else None,
            "softmax_weight": alpha,
            "hier_aux_prob_weight": float(hier_aux_prob_weight) if hier_aux_prob_weight is not None else None,
            "fusion_kind": fusion_mode_from_alpha(alpha) if alpha is not None else None,
        }
    raise ValueError(f"Unknown calibration method: {method}")


def topk_labels(probs: np.ndarray, labels: Sequence[str], k: int = 2) -> list[str]:
    top = np.argsort(-np.asarray(probs), axis=1)[:, :k]
    return ["|".join(str(labels[i]) for i in row) for row in top]


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true, dtype=np.int64)
    probs = normalize_probabilities(probs)
    y_pred = probs.argmax(axis=1)
    top2 = np.argsort(-probs, axis=1)[:, : min(2, probs.shape[1])]
    top2_ok = np.array([yt in row for yt, row in zip(y_true, top2)], dtype=bool)

    return {
        "n": int(len(y_true)),
        "top1_acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(len(labels))),
                average="macro",
                zero_division=0,
            )
        ),
        "top2_acc": float(top2_ok.mean()) if len(top2_ok) else 0.0,
    }


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = normalize_probabilities(probs)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(float(bin_acc) - float(bin_conf))

    one_hot = np.zeros_like(probs, dtype=np.float32)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    true_prob = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    nll = -np.mean(np.log(true_prob))

    return {
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll),
    }


def coverage_risk_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    thresholds: Iterable[float],
) -> list[dict[str, float | None]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=np.float32)
    rows: list[dict[str, float | None]] = []

    for threshold in thresholds:
        accepted = confidence >= float(threshold)
        n_accept = int(accepted.sum())
        coverage = float(n_accept / len(confidence)) if len(confidence) else 0.0
        if n_accept == 0:
            selective_risk = None
            selective_acc = None
        else:
            selective_acc = float((y_true[accepted] == y_pred[accepted]).mean())
            selective_risk = float(1.0 - selective_acc)
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "selective_risk": selective_risk,
                "selective_acc": selective_acc,
                "n_accepted": n_accept,
                "n_total": int(len(confidence)),
            }
        )
    return rows


def select_threshold_for_target_coverage(confidence: np.ndarray, target_coverage: float) -> float:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    conf = np.asarray(confidence, dtype=np.float32)
    if len(conf) == 0:
        raise ValueError("Cannot select threshold from an empty confidence array.")
    q = max(0.0, min(1.0, 1.0 - target_coverage))
    return float(np.quantile(conf, q))


def select_per_class_thresholds(
    y_pred: np.ndarray,
    confidence: np.ndarray,
    labels: Sequence[str],
    target_coverage: float,
    min_count: int,
    fallback_threshold: float,
) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float32)
    out: dict[str, float] = {}
    for idx, label in enumerate(labels):
        vals = confidence[y_pred == idx]
        if len(vals) >= min_count:
            out[str(label)] = select_threshold_for_target_coverage(vals, target_coverage)
        else:
            out[str(label)] = float(fallback_threshold)
    return out


def prediction_frame(
    metadata: pd.DataFrame,
    y_main: np.ndarray,
    y_aux: np.ndarray,
    main_labels: Sequence[str],
    aux_labels: Sequence[str],
    probs_by_method: Mapping[str, np.ndarray],
    aux_probs: np.ndarray | None = None,
    aux_proto_probs: np.ndarray | None = None,
    prototype_scores: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    frame = metadata.copy()
    frame["y_true_id"] = np.asarray(y_main, dtype=np.int64)
    frame["y_true"] = [main_labels[i] for i in y_main]
    frame["aux_true_id"] = np.asarray(y_aux, dtype=np.int64)
    frame["aux_true"] = [aux_labels[i] for i in y_aux]

    for method, probs in probs_by_method.items():
        p = normalize_probabilities(probs)
        pred = p.argmax(axis=1)
        frame[f"{method}_pred_id"] = pred
        frame[f"{method}_pred"] = [main_labels[i] for i in pred]
        frame[f"{method}_confidence"] = p.max(axis=1)
        frame[f"{method}_top2"] = topk_labels(p, main_labels, k=2)
        for i, label in enumerate(main_labels):
            frame[f"{method}_prob_{label}"] = p[:, i]

    if aux_probs is not None:
        aux_p = normalize_probabilities(aux_probs)
        aux_pred = aux_p.argmax(axis=1)
        frame["softmax_aux_pred_id"] = aux_pred
        frame["softmax_aux_pred"] = [aux_labels[i] for i in aux_pred]
        frame["softmax_aux_top2"] = topk_labels(aux_p, aux_labels, k=2)
        for i, label in enumerate(aux_labels):
            frame[f"softmax_aux_prob_{label}"] = aux_p[:, i]

    if aux_proto_probs is not None:
        aux_proto_p = normalize_probabilities(aux_proto_probs)
        proto_aux_pred = aux_proto_p.argmax(axis=1)
        frame["prototype_aux_pred_id"] = proto_aux_pred
        frame["prototype_aux_pred"] = [aux_labels[i] for i in proto_aux_pred]
        frame["prototype_aux_confidence"] = aux_proto_p.max(axis=1)
        frame["prototype_aux_top2"] = topk_labels(aux_proto_p, aux_labels, k=2)
        for i, label in enumerate(aux_labels):
            frame[f"prototype_aux_prob_{label}"] = aux_proto_p[:, i]

    if prototype_scores is not None:
        main_sim = np.asarray(prototype_scores["main_cosine_similarity"], dtype=np.float32)
        main_cos = np.asarray(prototype_scores["main_cosine_distance"], dtype=np.float32)
        main_euc = np.asarray(prototype_scores["main_euclidean_distance"], dtype=np.float32)
        aux_sim = np.asarray(prototype_scores["aux_cosine_similarity"], dtype=np.float32)
        aux_cos = np.asarray(prototype_scores["aux_cosine_distance"], dtype=np.float32)
        aux_euc = np.asarray(prototype_scores["aux_euclidean_distance"], dtype=np.float32)

        nearest_main = np.asarray(prototype_scores["nearest_main_id"], dtype=np.int64)
        nearest_aux = np.asarray(prototype_scores["nearest_aux_id"], dtype=np.int64)
        frame["nearest_main_prototype_id"] = nearest_main
        frame["nearest_main_prototype"] = [main_labels[i] for i in nearest_main]
        frame["nearest_main_cosine_similarity"] = main_sim[np.arange(len(frame)), nearest_main]
        frame["nearest_main_cosine_distance"] = main_cos[np.arange(len(frame)), nearest_main]
        frame["nearest_main_euclidean_distance"] = main_euc[np.arange(len(frame)), nearest_main]
        frame["nearest_aux_prototype_id"] = nearest_aux
        frame["nearest_aux_prototype"] = [aux_labels[i] for i in nearest_aux]
        frame["nearest_aux_cosine_similarity"] = aux_sim[np.arange(len(frame)), nearest_aux]
        frame["nearest_aux_cosine_distance"] = aux_cos[np.arange(len(frame)), nearest_aux]
        frame["nearest_aux_euclidean_distance"] = aux_euc[np.arange(len(frame)), nearest_aux]

        for i, label in enumerate(main_labels):
            frame[f"main_proto_cosine_similarity_{label}"] = main_sim[:, i]
            frame[f"main_proto_cosine_distance_{label}"] = main_cos[:, i]
            frame[f"main_proto_euclidean_distance_{label}"] = main_euc[:, i]
        for i, label in enumerate(aux_labels):
            frame[f"aux_proto_cosine_similarity_{label}"] = aux_sim[:, i]
            frame[f"aux_proto_cosine_distance_{label}"] = aux_cos[:, i]
            frame[f"aux_proto_euclidean_distance_{label}"] = aux_euc[:, i]

    if {"prototype_pred", "prototype_aux_pred"}.issubset(frame.columns):
        inconsistent = hierarchy_inconsistency_flags(
            frame["prototype_pred"].astype(str).tolist(),
            frame["prototype_aux_pred"].astype(str).tolist(),
        )
        frame["hierarchy_inconsistent"] = inconsistent
        frame["prototype_aux_mapped_main"] = [
            map_aux_label_to_main(x) for x in frame["prototype_aux_pred"].astype(str).tolist()
        ]

    return frame.copy()


def extract_embeddings(
    model,
    dataset,
    device: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings: list[np.ndarray] = []
    main_probs: list[np.ndarray] = []
    aux_probs: list[np.ndarray] = []
    y_main_all: list[np.ndarray] = []
    y_aux_all: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for xb, y_main, y_aux in loader:
            xb = xb.to(device)
            z = model.encode(xb)
            main_logits = model.main_head(z)
            aux_logits = model.aux_head(z)

            embeddings.append(z.detach().cpu().numpy().astype(np.float32))
            main_probs.append(torch.softmax(main_logits, dim=1).detach().cpu().numpy().astype(np.float32))
            aux_probs.append(torch.softmax(aux_logits, dim=1).detach().cpu().numpy().astype(np.float32))
            y_main_all.append(y_main.detach().cpu().numpy().astype(np.int64))
            y_aux_all.append(y_aux.detach().cpu().numpy().astype(np.int64))

    meta = dataset.df.copy().reset_index(drop=True)
    if dataset.path_col != "path":
        meta["path"] = meta[dataset.path_col]
    if dataset.label_col != "label":
        meta["label"] = meta[dataset.label_col]
    meta["aux_label"] = meta["__aux_label"]

    return {
        "metadata": meta,
        "embeddings": np.concatenate(embeddings, axis=0),
        "softmax_probs": np.concatenate(main_probs, axis=0),
        "aux_softmax_probs": np.concatenate(aux_probs, axis=0),
        "y_main": np.concatenate(y_main_all, axis=0),
        "y_aux": np.concatenate(y_aux_all, axis=0),
    }


def compute_feature_stats(
    dataset,
    num_main_classes: int,
    num_aux_classes: int | None = None,
) -> dict[str, np.ndarray]:
    sums: np.ndarray | None = None
    sums_sq: np.ndarray | None = None
    aux_sums: np.ndarray | None = None
    aux_sums_sq: np.ndarray | None = None
    counts = np.zeros(num_main_classes, dtype=np.int64)
    aux_counts = np.zeros(num_aux_classes, dtype=np.int64) if num_aux_classes is not None else None

    for idx in range(len(dataset)):
        x, y_main, y_aux = dataset[idx]
        arr = x.detach().cpu().numpy().astype(np.float64)
        class_id = int(y_main.item())
        if sums is None:
            shape = (num_main_classes,) + arr.shape
            sums = np.zeros(shape, dtype=np.float64)
            sums_sq = np.zeros(shape, dtype=np.float64)
            if num_aux_classes is not None:
                aux_shape = (num_aux_classes,) + arr.shape
                aux_sums = np.zeros(aux_shape, dtype=np.float64)
                aux_sums_sq = np.zeros(aux_shape, dtype=np.float64)
        sums[class_id] += arr
        sums_sq[class_id] += arr * arr
        counts[class_id] += 1
        if aux_counts is not None and aux_sums is not None and aux_sums_sq is not None:
            aux_id = int(y_aux.item())
            aux_sums[aux_id] += arr
            aux_sums_sq[aux_id] += arr * arr
            aux_counts[aux_id] += 1

    if sums is None or sums_sq is None:
        raise RuntimeError("Cannot compute feature stats from an empty dataset.")

    denom = np.maximum(counts.reshape((-1,) + (1,) * (sums.ndim - 1)), 1)
    mean = sums / denom
    var = np.maximum(sums_sq / denom - mean * mean, 0.0)
    std = np.sqrt(var)

    out = {
        "feature_main_mean": mean.astype(np.float32),
        "feature_main_std": std.astype(np.float32),
        "feature_main_counts": counts,
    }
    if aux_counts is not None and aux_sums is not None and aux_sums_sq is not None:
        aux_denom = np.maximum(aux_counts.reshape((-1,) + (1,) * (aux_sums.ndim - 1)), 1)
        aux_mean = aux_sums / aux_denom
        aux_var = np.maximum(aux_sums_sq / aux_denom - aux_mean * aux_mean, 0.0)
        out.update(
            {
                "feature_aux_mean": aux_mean.astype(np.float32),
                "feature_aux_std": np.sqrt(aux_var).astype(np.float32),
                "feature_aux_counts": aux_counts,
            }
        )
    return out


def save_prototype_bundle(
    out_path: str | Path,
    *,
    main_prototypes: np.ndarray,
    aux_prototypes: np.ndarray,
    main_counts: np.ndarray,
    aux_counts: np.ndarray,
    main_labels: Sequence[str],
    aux_labels: Sequence[str],
    metadata: Mapping[str, Any],
    feature_stats: Mapping[str, np.ndarray] | None = None,
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "main_prototypes": np.asarray(main_prototypes, dtype=np.float32),
        "aux_prototypes": np.asarray(aux_prototypes, dtype=np.float32),
        "main_counts": np.asarray(main_counts, dtype=np.int64),
        "aux_counts": np.asarray(aux_counts, dtype=np.int64),
        "main_labels": np.asarray(list(main_labels)),
        "aux_labels": np.asarray(list(aux_labels)),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    if feature_stats:
        arrays.update({k: np.asarray(v) for k, v in feature_stats.items()})
    np.savez_compressed(p, **arrays)


def load_prototype_bundle(path: str | Path) -> dict[str, Any]:
    p = require_file(path, "prototype bundle")
    data = np.load(p, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    out["main_labels"] = [str(x) for x in out["main_labels"].tolist()]
    out["aux_labels"] = [str(x) for x in out["aux_labels"].tolist()]
    out["metadata"] = json.loads(str(out["metadata_json"].tolist()))
    return out


def prototype_scores_from_bundle(
    embeddings: np.ndarray,
    bundle: Mapping[str, Any],
    temperature: float,
    hier_aux_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    main_sim = cosine_similarities(embeddings, np.asarray(bundle["main_prototypes"], dtype=np.float32))
    aux_sim = cosine_similarities(embeddings, np.asarray(bundle["aux_prototypes"], dtype=np.float32))
    main_cos_dist = (1.0 - main_sim).astype(np.float32)
    aux_cos_dist = (1.0 - aux_sim).astype(np.float32)
    main_euc_dist = euclidean_distances(embeddings, np.asarray(bundle["main_prototypes"], dtype=np.float32))
    aux_euc_dist = euclidean_distances(embeddings, np.asarray(bundle["aux_prototypes"], dtype=np.float32))
    main_proto_probs = similarity_to_probabilities(main_sim, temperature=temperature)
    aux_proto_probs = similarity_to_probabilities(aux_sim, temperature=temperature)
    hier_probs = hierarchical_main_probabilities(
        main_proto_probs,
        aux_proto_probs,
        bundle["aux_labels"],
        bundle["main_labels"],
        aux_weight=hier_aux_weight,
    )
    return {
        "prototype": main_proto_probs,
        "aux_prototype": aux_proto_probs,
        "hierarchical": hier_probs,
        "main_cosine_similarity": main_sim.astype(np.float32),
        "aux_cosine_similarity": aux_sim.astype(np.float32),
        "main_cosine_distance": main_cos_dist,
        "aux_cosine_distance": aux_cos_dist,
        "main_euclidean_distance": main_euc_dist,
        "aux_euclidean_distance": aux_euc_dist,
        "nearest_main_id": main_sim.argmax(axis=1).astype(np.int64),
        "nearest_aux_id": aux_sim.argmax(axis=1).astype(np.int64),
    }


def prototype_probabilities_from_bundle(
    embeddings: np.ndarray,
    bundle: Mapping[str, Any],
    temperature: float,
    hier_aux_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    return prototype_scores_from_bundle(
        embeddings,
        bundle,
        temperature=temperature,
        hier_aux_weight=hier_aux_weight,
    )


def method_metrics_table(
    y_true: np.ndarray,
    probs_by_method: Mapping[str, np.ndarray],
    labels: Sequence[str],
    n_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, probs in probs_by_method.items():
        row: dict[str, Any] = {"method": method}
        row.update(multiclass_metrics(y_true, probs, labels))
        row.update(calibration_metrics(y_true, probs, n_bins=n_bins))
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchy_consistency_rate(
    main_pred: Sequence[str],
    aux_pred: Sequence[str],
) -> float:
    if len(main_pred) == 0:
        return 0.0
    mapped = [map_aux_label_to_main(x) for x in aux_pred]
    ok = [str(a) == str(b) for a, b in zip(main_pred, mapped)]
    return float(np.mean(ok))


def align_prediction_frames_by_path(
    reference: pd.DataFrame,
    reproduced: pd.DataFrame,
    path_col: str = "path",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if path_col not in reference.columns:
        raise ValueError(f"reference predictions are missing required path column: {path_col}")
    if path_col not in reproduced.columns:
        raise ValueError(f"reproduced predictions are missing required path column: {path_col}")

    ref = reference.copy()
    rep = reproduced.copy()
    ref["_norm_path"] = [normalize_identity_path(x) for x in ref[path_col].tolist()]
    rep["_norm_path"] = [normalize_identity_path(x) for x in rep[path_col].tolist()]

    ref_duplicates = sorted(ref.loc[ref["_norm_path"].duplicated(keep=False), "_norm_path"].unique().tolist())
    rep_duplicates = sorted(rep.loc[rep["_norm_path"].duplicated(keep=False), "_norm_path"].unique().tolist())

    ref_paths = set(ref["_norm_path"])
    rep_paths = set(rep["_norm_path"])
    matched_paths = sorted(ref_paths & rep_paths)
    missing = sorted(ref_paths - rep_paths)
    extra = sorted(rep_paths - ref_paths)

    rows: list[dict[str, Any]] = []
    ref_idx = ref.drop_duplicates("_norm_path", keep="first").set_index("_norm_path")
    rep_idx = rep.drop_duplicates("_norm_path", keep="first").set_index("_norm_path")
    for path in sorted(ref_paths | rep_paths):
        in_ref = path in ref_idx.index
        in_rep = path in rep_idx.index
        row: dict[str, Any] = {
            "path": path,
            "in_reference": bool(in_ref),
            "in_reproduced": bool(in_rep),
            "duplicate_in_reference": path in ref_duplicates,
            "duplicate_in_reproduced": path in rep_duplicates,
        }
        if in_ref:
            row["reference_y_true"] = ref_idx.at[path, "y_true"] if "y_true" in ref_idx.columns else None
            row["reference_y_pred"] = ref_idx.at[path, "y_pred"] if "y_pred" in ref_idx.columns else None
        else:
            row["reference_y_true"] = None
            row["reference_y_pred"] = None
        if in_rep:
            row["reproduced_y_true"] = rep_idx.at[path, "y_true"] if "y_true" in rep_idx.columns else None
            row["reproduced_y_pred"] = rep_idx.at[path, "y_pred"] if "y_pred" in rep_idx.columns else None
        else:
            row["reproduced_y_true"] = None
            row["reproduced_y_pred"] = None
        row["y_true_match"] = bool(in_ref and in_rep and str(row["reference_y_true"]) == str(row["reproduced_y_true"]))
        row["y_pred_match"] = bool(in_ref and in_rep and str(row["reference_y_pred"]) == str(row["reproduced_y_pred"]))
        rows.append(row)

    report = pd.DataFrame(rows)
    matched = report[report["in_reference"] & report["in_reproduced"]]
    summary = {
        "alignment_ok": bool(
            not missing
            and not extra
            and not ref_duplicates
            and not rep_duplicates
            and bool(matched["y_true_match"].all()) if len(matched) else False
        ),
        "matched_count": int(len(matched)),
        "reference_count": int(len(ref)),
        "reproduced_count": int(len(rep)),
        "missing_reference_paths": missing,
        "extra_reproduced_paths": extra,
        "duplicate_reference_paths": ref_duplicates,
        "duplicate_reproduced_paths": rep_duplicates,
        "y_true_match_count": int(matched["y_true_match"].sum()) if len(matched) else 0,
        "y_pred_match_count": int(matched["y_pred_match"].sum()) if len(matched) else 0,
        "y_true_all_match": bool(len(matched) > 0 and matched["y_true_match"].all() and not missing and not extra),
        "y_pred_all_match": bool(len(matched) > 0 and matched["y_pred_match"].all() and not missing and not extra),
    }
    return report, summary


def audit_manifest_disjointness(
    manifest_paths: Mapping[str, str | Path],
    columns: Sequence[str] = ("path", "source_id", "md5"),
) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    required = list(columns)
    missing_columns: dict[str, list[str]] = {}
    invalid_values: list[dict[str, Any]] = []
    normalized: dict[str, dict[str, list[str]]] = {}

    def invalid_reason(column: str, raw: Any, normalized_value: str) -> str | None:
        if raw is None:
            return "null"
        if pd.isna(raw):
            return "nan"
        text = str(raw).strip()
        if text == "":
            return "empty"
        if text.lower() in {"nan", "none"}:
            return text.lower()
        if column == "md5" and not re.fullmatch(r"[0-9a-f]{32}", normalized_value):
            return "invalid_md5"
        return None

    def normalize(column: str, raw: Any) -> str:
        if column == "path":
            return normalize_identity_path(raw)
        if column == "source_id":
            return normalize_source_id(raw)
        if column == "md5":
            return normalize_md5(raw)
        return str(raw).strip()

    for split, path in manifest_paths.items():
        p = require_file(path, f"{split} manifest")
        df = pd.read_csv(p)
        split_name = str(split)
        frames[split_name] = df
        counts[split_name] = int(len(df))
        missing = [column for column in required if column not in df.columns]
        missing_columns[split_name] = missing
        normalized[split_name] = {}
        for column in required:
            if column in missing:
                continue
            values: list[str] = []
            for row_idx, raw in enumerate(df[column].tolist()):
                norm = normalize(column, raw)
                reason = invalid_reason(column, raw, norm)
                if reason:
                    invalid_values.append(
                        {
                            "split": split_name,
                            "row": int(row_idx),
                            "column": column,
                            "reason": reason,
                            "value": None if raw is None or pd.isna(raw) else str(raw),
                        }
                    )
                values.append(norm)
            normalized[split_name][column] = values

    overlaps: list[dict[str, Any]] = []
    splits = list(frames.keys())
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            for column in required:
                if column not in normalized[left] or column not in normalized[right]:
                    continue
                left_vals = set(normalized[left][column])
                right_vals = set(normalized[right][column])
                shared = sorted(left_vals & right_vals)
                if shared:
                    overlaps.append(
                        {
                            "left": left,
                            "right": right,
                            "column": column,
                            "count": int(len(shared)),
                            "examples": shared[:5],
                        }
                    )

    all_columns_checked = all(not vals for vals in missing_columns.values())
    return {
        "ok": all_columns_checked and len(invalid_values) == 0 and len(overlaps) == 0,
        "counts": counts,
        "columns_checked": required if all_columns_checked else [],
        "missing_columns": missing_columns,
        "invalid_values": invalid_values,
        "overlaps": overlaps,
    }


def assert_no_leakage(report: Mapping[str, Any]) -> None:
    if not bool(report.get("ok", False)):
        raise RuntimeError(
            "Manifest leakage detected: "
            f"missing_columns={report.get('missing_columns')}; "
            f"invalid_values={report.get('invalid_values')}; "
            f"overlaps={report.get('overlaps')}"
        )


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
