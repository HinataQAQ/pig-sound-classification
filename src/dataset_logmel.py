# src/dataset_logmel.py
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

from .utils.audio_io import load_wav, fix_length


def compute_logmel(
    y, sr,
    n_mels=64, fmin=50, fmax=8000,
    hop_ms=10.0, win_ms=25.0, n_fft=1024,
    ref_mode="fixed", db_ref=1.0, top_db=80.0,
    norm="01",   # "01" | "cmvn" | "none"
):
    """
    返回 (T, F) float32

    ref_mode:
      - "fixed": ref = db_ref (标量) -> 保留绝对能量差（推荐 cough vs silence）
      - "max":   ref = np.max        -> 每条样本按自身峰值归一（会削弱能量差）

    norm:
      - "01":   把 dB 范围 [-top_db, 0] -> [0, 1]（推荐）
      - "cmvn": 逐条 CMVN（会削弱能量差）
      - "none": 不额外归一化（直接 dB）
    """
    sr = int(sr)
    hop_length = int(round(sr * float(hop_ms) / 1000.0))
    win_length = int(round(sr * float(win_ms) / 1000.0))

    n_fft = int(n_fft)
    if n_fft < win_length:
        n_fft = 1 << (win_length - 1).bit_length()

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
        n_mels=int(n_mels), fmin=float(fmin), fmax=float(fmax),
        power=2.0, center=False
    )

    if str(ref_mode).lower() in ("max", "peak"):
        ref = np.max
    else:
        ref = float(db_ref)

    # power_to_db 支持 ref 是标量或 callable（例如 np.max） :contentReference[oaicite:2]{index=2}
    logmel = librosa.power_to_db(mel, ref=ref, top_db=float(top_db)).T  # (T,F)

    norm = str(norm).lower()
    if norm == "01":
        # [-top_db, 0] -> [0, 1]
        logmel = (logmel + float(top_db)) / float(top_db)
        logmel = np.clip(logmel, 0.0, 1.0)
    elif norm == "cmvn":
        mu = logmel.mean(axis=0, keepdims=True)
        std = logmel.std(axis=0, keepdims=True) + 1e-6
        logmel = (logmel - mu) / std
    else:
        # "none"
        pass

    return logmel.astype(np.float32)


def spec_augment_tf_mask(
    feat: np.ndarray,
    rng: np.random.Generator,
    time_ratio: float = 0.05,
    freq_ratio: float = 0.05,
    num_time_masks: int = 1,
    num_freq_masks: int = 1,
    mask_value: float = 0.0,
) -> np.ndarray:
    """简化 SpecAugment：Time mask + Freq mask（对 log-mel 直接做遮挡）"""
    T, F = feat.shape
    out = feat.copy()

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
        manifest_path: str,
        labels,
        sr: int = 32000,
        segment_seconds: float = 1.0,
        n_mels: int = 64,
        fmin: int = 50,
        fmax: int = 8000,
        hop_ms: float = 10.0,
        win_ms: float = 25.0,
        n_fft: int = 1024,
        use_specaug: bool = False,
        seed: int = 3407,
        train: bool = True,
        ref_mode="fixed",
        db_ref=1.0,
        top_db=80.0,
        norm="01",
    ):
        self.df = pd.read_csv(manifest_path)
        self.df["filepath"] = self.df["filepath"].astype(str)
        self.df["label"] = self.df["label"].astype(str).str.strip().str.lower()

        self.labels = [str(l).strip().lower() for l in labels]
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
        self.rng = np.random.default_rng(int(seed))
        self.train = bool(train)

        self.ref_mode = ref_mode
        self.db_ref = float(db_ref)
        self.top_db = float(top_db)
        self.norm = norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = Path(row["filepath"])
        lab = row["label"]
        if lab not in self.label2id:
            raise KeyError(f"label '{lab}' not in config labels {self.labels}")
        y_id = self.label2id[lab]

        y, _ = load_wav(str(wav_path), sr=self.sr, mono=True)

        # 固定长度：训练随机裁剪，验证取中间
        if len(y) < self.seg_len:
            y = fix_length(y, self.seg_len)
        else:
            if self.train:
                start = random.randint(0, len(y) - self.seg_len)
            else:
                start = max(0, (len(y) - self.seg_len) // 2)
            y = y[start:start + self.seg_len]

        feat = compute_logmel(
            y, sr=self.sr,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            hop_ms=self.hop_ms, win_ms=self.win_ms, n_fft=self.n_fft,
            ref_mode=self.ref_mode, db_ref=self.db_ref, top_db=self.top_db,
            norm=self.norm,
        )

        if self.train and self.use_specaug:
            # norm="01" 时 mask_value=0 表示“更安静”，合理
            feat = spec_augment_tf_mask(feat, self.rng, mask_value=0.0)

        x = torch.from_numpy(feat)          # (T,F)
        ylab = torch.tensor(y_id).long()
        return x, ylab
