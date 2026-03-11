# src/dataset_logmel.py
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

from .utils.audio_io import load_wav, fix_length
from .preprocessing.denoise import spectral_subtraction


def compute_logmel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    fmin: float = 50,
    fmax: float = 8000,
    hop_ms: float = 10.0,
    win_ms: float = 25.0,
    n_fft: int = 1024,
    ref_mode: str = "fixed",   # "fixed" 保留能量差；"max" 会把每条音频都归一到 0dB
    db_ref: float = 1.0,
    top_db: float = 80.0,
) -> np.ndarray:
    hop_length = int(sr * hop_ms / 1000.0)
    win_length = int(sr * win_ms / 1000.0)

    if n_fft < win_length:
        # 保证 n_fft >= win_length
        n_fft = 1 << (win_length - 1).bit_length()

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

    rm = str(ref_mode).lower()
    if rm in ("max", "peak"):
        ref = np.max
    else:
        ref = float(db_ref)

    logmel = librosa.power_to_db(mel, ref=ref, top_db=float(top_db)).T  # (T,F)
    return logmel.astype(np.float32)


def spec_augment_tf_mask(
    feat: np.ndarray,
    rng: np.random.Generator,
    time_ratio: float = 0.05,
    freq_ratio: float = 0.05,
    num_time_masks: int = 1,
    num_freq_masks: int = 1,
    mask_value: Optional[float] = None,
) -> np.ndarray:
    """简化 SpecAugment：Time mask + Freq mask（对 log-mel 做遮挡）"""
    T, F = feat.shape
    out = feat.copy()

    if mask_value is None:
        mask_value = float(out.min())  # 对未归一化 logmel，用最小值（接近 -80dB）更合理

    tmax = max(1, int(round(time_ratio * T)))
    fmax = max(1, int(round(freq_ratio * F)))

    for _ in range(num_time_masks):
        t = int(rng.integers(0, tmax + 1))
        if t <= 0 or T - t <= 0:
            continue
        t0 = int(rng.integers(0, T - t + 1))
        out[t0:t0 + t, :] = mask_value

    for _ in range(num_freq_masks):
        f = int(rng.integers(0, fmax + 1))
        if f <= 0 or F - f <= 0:
            continue
        f0 = int(rng.integers(0, F - f + 1))
        out[:, f0:f0 + f] = mask_value

    return out


class LogMelDataset(Dataset):
    """
    manifests(csv: filepath,label) -> 输出 log-mel (T,F) + label_id
    """
    def __init__(
        self,
        manifest_path: Optional[str] = None,
        labels=None,
        sr: int = 32000,
        segment_seconds: float = 1.0,
        n_mels: int = 64,
        fmin: int = 50,
        fmax: int = 8000,
        hop_ms: float = 10.0,
        win_ms: float = 25.0,
        n_fft: int = 1024,
        use_specaug: bool = False,
        specaug: Optional[Dict[str, Any]] = None,
        seed: int = 3407,
        train: bool = True,
        do_denoise: bool = False,
        ref_mode: str = "fixed",
        db_ref: float = 1.0,
        top_db: float = 80.0,
        cmvn: str = "none",            # "none" / "utt"
        return_path: bool = False,
        **kwargs,  # 兼容你旧代码里可能传进来的多余参数，避免 TypeError
    ):
        if manifest_path is None:
            raise ValueError("manifest_path is required")

        self.df = pd.read_csv(manifest_path)
        self.df["filepath"] = self.df["filepath"].astype(str)
        self.df["label"] = self.df["label"].astype(str).str.strip().str.lower()

        self.labels = [str(l).strip().lower() for l in (labels or [])]
        self.label2id = {l: i for i, l in enumerate(self.labels)}

        self.sr = int(sr)
        self.seg_len = int(round(self.sr * float(segment_seconds)))

        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.hop_ms = float(hop_ms)
        self.win_ms = float(win_ms)
        self.n_fft = int(n_fft)

        self.use_specaug = bool(use_specaug)
        self.specaug = specaug or {}
        self.rng = np.random.default_rng(int(seed))
        self.train = bool(train)

        self.do_denoise = bool(do_denoise)

        self.ref_mode = str(ref_mode)
        self.db_ref = float(db_ref)
        self.top_db = float(top_db)

        self.cmvn = str(cmvn).lower()
        self.return_path = bool(return_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = str(Path(row["filepath"]))
        lab = str(row["label"]).strip().lower()
        if lab not in self.label2id:
            raise KeyError(f"label '{lab}' not in config labels {self.labels}")
        y_id = self.label2id[lab]

        y, _sr = load_wav(wav_path, sr=self.sr, mono=True)

        if self.do_denoise:
            y = spectral_subtraction(y, self.sr)

        # 固定长度：训练随机裁剪，验证/测试取中间
        if len(y) < self.seg_len:
            y = fix_length(y, self.seg_len)
        else:
            if self.train:
                start = int(self.rng.integers(0, len(y) - self.seg_len + 1))
            else:
                start = max(0, (len(y) - self.seg_len) // 2)
            y = y[start:start + self.seg_len]

        feat = compute_logmel(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_ms=self.hop_ms,
            win_ms=self.win_ms,
            n_fft=self.n_fft,
            ref_mode=self.ref_mode,
            db_ref=self.db_ref,
            top_db=self.top_db,
        )  # (T,F)

        # 可选：逐条 CMVN（注意：cough vs silence 想冲 90%，一般先关掉）
        if self.cmvn == "utt":
            mu = feat.mean(axis=0, keepdims=True)
            std = feat.std(axis=0, keepdims=True) + 1e-6
            feat = (feat - mu) / std

        if self.train and self.use_specaug:
            feat = spec_augment_tf_mask(
                feat,
                self.rng,
                time_ratio=float(self.specaug.get("time_ratio", 0.05)),
                freq_ratio=float(self.specaug.get("freq_ratio", 0.05)),
                num_time_masks=int(self.specaug.get("num_time_masks", 1)),
                num_freq_masks=int(self.specaug.get("num_freq_masks", 1)),
                mask_value=(0.0 if self.cmvn == "utt" else None),
            )

        x = torch.from_numpy(feat)           # (T,F)
        ylab = torch.tensor(y_id).long()

        if self.return_path:
            return x, ylab, wav_path
        return x, ylab
