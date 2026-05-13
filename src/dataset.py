from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features.gfcc import add_deltas as gfcc_delta
from .features.gfcc import compute_gfcc
from .features.mfcc import add_deltas as mfcc_delta
from .features.mfcc import compute_mfcc
from .preprocessing.denoise import spectral_subtraction
from .preprocessing.vad import crop_around_activity, fixed_length_crop
from .utils.audio_io import load_wav


class PigAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        labels,
        sr: int = 16000,
        segment_seconds: float = 1.0,
        do_denoise: bool = True,
        do_vad: bool = False,
        include_delta: bool = True,
        include_delta_delta: bool = False,
        train: bool = True,
        mfcc_config: Optional[Dict] = None,
        gfcc_config: Optional[Dict] = None,
        vad_config: Optional[Dict] = None,
        cmvn: str = "utt",
    ):
        self.df = pd.read_csv(manifest_path)
        self.df["filepath"] = self.df["filepath"].astype(str)
        self.df["label"] = self.df["label"].astype(str).str.strip().str.lower()

        self.labels = [str(l).strip().lower() for l in labels]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        self.sr = int(sr)
        self.seg_len = int(round(self.sr * float(segment_seconds)))
        self.do_denoise = bool(do_denoise)
        self.do_vad = bool(do_vad)
        self.include_delta = bool(include_delta)
        self.include_delta_delta = bool(include_delta_delta)
        self.train = bool(train)

        self.mfcc_config = mfcc_config or {}
        self.gfcc_config = gfcc_config or {}
        self.vad_config = vad_config or {}
        self.cmvn = str(cmvn).lower()

    def __len__(self):
        return len(self.df)

    def _crop(self, y: np.ndarray) -> np.ndarray:
        if self.do_vad:
            y, _meta = crop_around_activity(
                y,
                sr=self.sr,
                target_len=self.seg_len,
                train=self.train,
                **self.vad_config,
            )
            return y.astype(np.float32)

        y, _start = fixed_length_crop(y, target_len=self.seg_len, train=self.train)
        return y.astype(np.float32)

    def __getitem__(self, idx):
        hop_len = int(round(0.010 * self.sr))
        win_len = int(round(0.025 * self.sr))
        hop_time = hop_len / float(self.sr)
        window_time = win_len / float(self.sr)

        row = self.df.iloc[idx]
        path = row["filepath"]
        raw_lab = str(row["label"])
        lab_norm = raw_lab.strip().lower()
        if lab_norm not in self.label2id:
            raise KeyError(
                f"Label '{raw_lab}' (normalized '{lab_norm}') not in config labels {self.labels}. "
                f"Fix config.data.labels to match manifests."
            )
        label = self.label2id[lab_norm]

        y, sr = load_wav(path, sr=self.sr, mono=True)
        if self.do_denoise:
            y = spectral_subtraction(y, sr)

        y = self._crop(y)

        mfcc = compute_mfcc(
            y,
            sr=self.sr,
            n_mfcc=int(self.mfcc_config.get("n_mfcc", 26)),
            n_mels=int(self.mfcc_config.get("n_mels", 64)),
            fmin=float(self.mfcc_config.get("fmin", 50)),
            fmax=float(self.mfcc_config.get("fmax", 8000)),
            hop_length=hop_len,
            win_length=win_len,
            center=False,
        )
        mfcc = mfcc_delta(mfcc, self.include_delta, self.include_delta_delta)

        gfcc = compute_gfcc(
            y,
            sr=self.sr,
            n_gfcc=int(self.gfcc_config.get("n_gfcc", 32)),
            n_filters=int(self.gfcc_config.get("n_filters", 64)),
            window_time=window_time,
            hop_time=hop_time,
            fmin=float(self.gfcc_config.get("fmin", 50)),
            fmax=float(self.gfcc_config.get("fmax", 8000)),
        )
        gfcc = gfcc_delta(gfcc, self.include_delta, self.include_delta_delta)

        T = min(mfcc.shape[0], gfcc.shape[0])
        mfcc = mfcc[:T]
        gfcc = gfcc[:T]
        feat = np.concatenate([mfcc, gfcc], axis=1).astype(np.float32)

        if self.cmvn == "utt":
            mu = feat.mean(axis=0, keepdims=True)
            std = feat.std(axis=0, keepdims=True) + 1e-6
            feat = (feat - mu) / std

        x = torch.from_numpy(feat)
        ylab = torch.tensor(label).long()
        return x, ylab
