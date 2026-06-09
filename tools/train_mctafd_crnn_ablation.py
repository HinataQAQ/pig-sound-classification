from pathlib import Path
from math import gcd
import argparse
import random
import json

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly, butter, sosfiltfilt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_path_col(df):
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column. columns={list(df.columns)}")


def infer_label_col(df):
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column. columns={list(df.columns)}")


def load_audio_soundfile(path, sr=32000, dur_s=1.0):
    """
    用 soundfile 读取音频，避免部分 wav 在 librosa.load 阶段卡住。
    """
    path = str(Path(path))

    y, s = sf.read(path, dtype="float32", always_2d=False)

    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = y.mean(axis=1)

    y = np.asarray(y, dtype=np.float32)

    if int(s) != int(sr):
        g = gcd(int(s), int(sr))
        up = int(sr) // g
        down = int(s) // g
        y = resample_poly(y, up, down).astype(np.float32)
        s = int(sr)

    target_len = int(sr * dur_s)

    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y.astype(np.float32), int(s)


def norm_map(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / (sd + eps)).astype(np.float32)

def rms_normalize(y, target_db=-20.0, eps=1e-8):
    """
    自动响度归一化，不需要人工听音。
    target_db=-20 表示把 RMS 调到约 -20 dBFS。
    """
    y = np.asarray(y, dtype=np.float32)

    rms = float(np.sqrt(np.mean(y * y) + eps))
    target = 10.0 ** (target_db / 20.0)

    if rms < eps:
        return y.astype(np.float32)

    y = y * (target / rms)
    y = np.clip(y, -1.0, 1.0)

    return y.astype(np.float32)


def highpass_filter(y, sr, cutoff=80.0, order=4):
    """
    去除低频轰鸣、风机低频、设备低频。
    """
    y = np.asarray(y, dtype=np.float32)

    if cutoff <= 0 or sr <= cutoff * 2:
        return y

    try:
        sos = butter(
            order,
            cutoff / (sr * 0.5),
            btype="highpass",
            output="sos",
        )
        out = sosfiltfilt(sos, y).astype(np.float32)
        return out
    except Exception:
        return y.astype(np.float32)


def spectral_gate_denoise(
    y,
    sr,
    n_fft=1024,
    hop_length=320,
    win_length=800,
    noise_percentile=20.0,
    strength=1.5,
    mask_power=1.5,
    blend=0.85,
):
    """
    简单谱门控降噪。
    不需要人工标注噪声片段。

    原理：
      1. 对整段音频做 STFT；
      2. 用每个频率上的低分位能量估计背景噪声；
      3. 低于噪声底的部分被抑制；
      4. 与原始音频做 blend，避免过度降噪破坏猪声。
    """
    y = np.asarray(y, dtype=np.float32)

    if len(y) < n_fft:
        return y

    try:
        S = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
        )

        mag = np.abs(S).astype(np.float32)
        phase = np.exp(1j * np.angle(S))

        noise = np.percentile(
            mag,
            q=noise_percentile,
            axis=1,
            keepdims=True,
        ).astype(np.float32)

        mask = (mag - strength * noise) / (mag + 1e-8)
        mask = np.clip(mask, 0.0, 1.0)
        mask = mask ** mask_power

        S_clean = (mag * mask) * phase

        y_clean = librosa.istft(
            S_clean,
            hop_length=hop_length,
            win_length=win_length,
            length=len(y),
        ).astype(np.float32)

        out = blend * y_clean + (1.0 - blend) * y
        out = np.clip(out, -1.0, 1.0)

        return out.astype(np.float32)

    except Exception:
        return y.astype(np.float32)


def preprocess_wave_for_logmel(
    y,
    sr,
    mode,
    n_fft=1024,
    hop_length=320,
    win_length=800,
):
    """
    mode:
      none  -> 原始音频
      hpf   -> 高通 + RMS 归一化
      sg    -> 高通 + RMS 归一化 + 谱门控降噪
    """
    y = np.asarray(y, dtype=np.float32)

    if mode == "none":
        return y

    if mode == "hpf":
        y = highpass_filter(y, sr=sr, cutoff=80.0)
        y = rms_normalize(y, target_db=-20.0)
        return y.astype(np.float32)

    if mode == "sg":
        y = highpass_filter(y, sr=sr, cutoff=80.0)
        y = rms_normalize(y, target_db=-20.0)
        y = spectral_gate_denoise(
            y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        y = rms_normalize(y, target_db=-20.0)
        return y.astype(np.float32)

    raise ValueError(f"Unknown preprocess mode: {mode}")


def make_logmel_or_pcen(
    y,
    sr=32000,
    n_mels=64,
    n_fft=1024,
    hop_length=320,
    win_length=800,
    fmin=50,
    fmax=8000,
    use_pcen=False,
):
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
    ).astype(np.float32)

    if use_pcen:
        feat = librosa.pcen(
            mel * (2.0 ** 31),
            sr=sr,
            hop_length=hop_length,
        ).astype(np.float32)
        return norm_map(feat)

    logmel = librosa.power_to_db(
        mel,
        ref=1.0,
        top_db=80.0,
    ).astype(np.float32)

    return norm_map(logmel)


def align_rows(mat, target_rows=64):
    """
    把 [R, T] 插值到 [64, T]，用于把 MFCC / Delta / Flux 对齐到 mel bins。
    """
    mat = np.asarray(mat, dtype=np.float32)
    r, t = mat.shape

    if r == target_rows:
        return mat.astype(np.float32)

    old_x = np.linspace(0.0, 1.0, r)
    new_x = np.linspace(0.0, 1.0, target_rows)

    out = np.zeros((target_rows, t), dtype=np.float32)

    for j in range(t):
        out[:, j] = np.interp(new_x, old_x, mat[:, j])

    return out.astype(np.float32)


def safe_delta(mfcc, order=1):
    """
    防止极短时间帧时 delta 报错。你的 1 秒音频一般 T 足够，但这里做防御。
    """
    t = mfcc.shape[1]
    width = 9

    if t < width:
        width = t if t % 2 == 1 else t - 1

    if width < 3:
        return np.zeros_like(mfcc, dtype=np.float32)

    return librosa.feature.delta(
        mfcc,
        width=width,
        order=order,
        axis=-1,
        mode="nearest",
    ).astype(np.float32)

def apply_specaugment(
    x,
    freq_masks=2,
    time_masks=2,
    freq_width=8,
    time_width=12,
):
    """
    x: torch.Tensor [B, C, F, T]
    只在 train 阶段调用。
    """
    if x.ndim != 4:
        return x

    b, c, f, t = x.shape

    x = x.clone()

    for i in range(b):
        for _ in range(freq_masks):
            if f <= 1:
                continue

            w = torch.randint(
                low=0,
                high=min(freq_width, f) + 1,
                size=(1,),
                device=x.device,
            ).item()

            if w <= 0:
                continue

            f0 = torch.randint(
                low=0,
                high=max(1, f - w + 1),
                size=(1,),
                device=x.device,
            ).item()

            x[i, :, f0:f0 + w, :] = 0.0

        for _ in range(time_masks):
            if t <= 1:
                continue

            w = torch.randint(
                low=0,
                high=min(time_width, t) + 1,
                size=(1,),
                device=x.device,
            ).item()

            if w <= 0:
                continue

            t0 = torch.randint(
                low=0,
                high=max(1, t - w + 1),
                size=(1,),
                device=x.device,
            ).item()

            x[i, :, :, t0:t0 + w] = 0.0

    return x

def make_feature(
    y,
    feature_mode,
    sr=32000,
    n_mels=64,
    n_mfcc=20,
    n_fft=1024,
    hop_length=320,
    win_length=800,
    fmin=50,
    fmax=8000,
):
    """
    原有 feature_mode:
      logmel
      mel_mfcc
      mel_mfcc_dyn
      mel_mfcc_dyn_flux
      mctafd

    新增 feature_mode:
      logmel_hpf       -> 高通 + RMS 归一化后的 Log-Mel
      logmel_sg        -> 高通 + RMS + 谱门控降噪后的 Log-Mel
      logmel_pcen      -> 原始音频的 PCEN
      logmel_sg_pcen   -> 降噪音频的 PCEN
      logmel_dual      -> 原始 Log-Mel + 降噪 Log-Mel
      logmel_dual_pcen -> 原始 Log-Mel + 降噪 Log-Mel + PCEN
    """

    # ===== 新增 Log-Mel 预处理路线 =====

    if feature_mode == "logmel_hpf":
        y_hpf = preprocess_wave_for_logmel(
            y,
            sr=sr,
            mode="hpf",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        ch = make_logmel_or_pcen(
            y_hpf,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=False,
        )

        return np.stack([ch], axis=0).astype(np.float32)

    if feature_mode == "logmel_sg":
        y_sg = preprocess_wave_for_logmel(
            y,
            sr=sr,
            mode="sg",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        ch = make_logmel_or_pcen(
            y_sg,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=False,
        )

        return np.stack([ch], axis=0).astype(np.float32)

    if feature_mode == "logmel_pcen":
        ch = make_logmel_or_pcen(
            y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=True,
        )

        return np.stack([ch], axis=0).astype(np.float32)

    if feature_mode == "logmel_sg_pcen":
        y_sg = preprocess_wave_for_logmel(
            y,
            sr=sr,
            mode="sg",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        ch = make_logmel_or_pcen(
            y_sg,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=True,
        )

        return np.stack([ch], axis=0).astype(np.float32)

    if feature_mode == "logmel_dual":
        raw = make_logmel_or_pcen(
            y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=False,
        )

        y_sg = preprocess_wave_for_logmel(
            y,
            sr=sr,
            mode="sg",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        sg = make_logmel_or_pcen(
            y_sg,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=False,
        )

        return np.stack([raw, sg], axis=0).astype(np.float32)

    if feature_mode == "logmel_dual_pcen":
        raw = make_logmel_or_pcen(
            y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=False,
        )

        y_sg = preprocess_wave_for_logmel(
            y,
            sr=sr,
            mode="sg",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        sg = make_logmel_or_pcen(
            y_sg,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=False,
        )

        pcen = make_logmel_or_pcen(
            y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            use_pcen=True,
        )

        return np.stack([raw, sg, pcen], axis=0).astype(np.float32)

    # ===== 原有路线保持不变 =====

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
    )

    logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)
    ch_logmel = norm_map(logmel)

    if feature_mode == "logmel":
        return np.stack([ch_logmel], axis=0).astype(np.float32)

    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc).astype(np.float32)
    mfcc_align = align_rows(mfcc, target_rows=n_mels)
    ch_mfcc = norm_map(mfcc_align)

    if feature_mode == "mel_mfcc":
        return np.stack([ch_logmel, ch_mfcc], axis=0).astype(np.float32)

    d1 = safe_delta(mfcc, order=1)
    d2 = safe_delta(mfcc, order=2)

    dyn = np.concatenate([d1, d2], axis=0)
    dyn_align = align_rows(dyn, target_rows=n_mels)
    ch_dyn = norm_map(dyn_align)

    if feature_mode == "mel_mfcc_dyn":
        return np.stack([ch_logmel, ch_mfcc, ch_dyn], axis=0).astype(np.float32)

    diff_t = np.diff(logmel, axis=1)
    flux = np.maximum(diff_t, 0.0)
    flux = np.concatenate([np.zeros((n_mels, 1), dtype=np.float32), flux], axis=1)
    ch_flux = norm_map(flux)

    if feature_mode == "mel_mfcc_dyn_flux":
        return np.stack([ch_logmel, ch_mfcc, ch_dyn, ch_flux], axis=0).astype(np.float32)

    if feature_mode == "mctafd":
        diff_map = np.abs(norm_map(logmel) - norm_map(mfcc_align)).astype(np.float32)
        ch_diff = norm_map(diff_map)

        return np.stack(
            [ch_logmel, ch_mfcc, ch_dyn, ch_flux, ch_diff],
            axis=0,
        ).astype(np.float32)

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def feature_channels(feature_mode):
    mapping = {
        "logmel": 1,
        "logmel_hpf": 1,
        "logmel_sg": 1,
        "logmel_pcen": 1,
        "logmel_sg_pcen": 1,
        "logmel_dual": 2,
        "logmel_dual_pcen": 3,

        "mel_mfcc": 2,
        "mel_mfcc_dyn": 3,
        "mel_mfcc_dyn_flux": 4,
        "mctafd": 5,
    }

    if feature_mode not in mapping:
        raise ValueError(f"Unknown feature_mode={feature_mode}")

    return mapping[feature_mode]


class PigVocalFeatureDataset(Dataset):
    def __init__(
        self,
        manifest,
        labels,
        feature_mode,
        sr=32000,
        dur_s=1.0,
        cache=True,
        n_mels=64,
        n_mfcc=20,
        n_fft=1024,
        hop_length=320,
        win_length=800,
        fmin=50,
        fmax=8000,
    ):
        self.df = pd.read_csv(manifest)
        self.path_col = infer_path_col(self.df)
        self.label_col = infer_label_col(self.df)

        self.labels = [x.strip().lower() for x in labels]
        self.label2id = {x: i for i, x in enumerate(self.labels)}

        self.feature_mode = feature_mode
        self.sr = sr
        self.dur_s = dur_s
        self.cache = cache
        self._cache = {}

        # Feature parameters
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax

        self.df[self.label_col] = self.df[self.label_col].astype(str).str.strip().str.lower()

        keep = self.df[self.label_col].isin(self.label2id.keys())
        bad = self.df[~keep]

        if len(bad) > 0:
            print("[WARN] dropping rows with unknown labels:")
            print(bad[self.label_col].value_counts())

        self.df = self.df[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        label = str(row[self.label_col]).strip().lower()

        y, sr = load_audio_soundfile(path, sr=self.sr, dur_s=self.dur_s)
        x = make_feature(
            y,
            feature_mode=self.feature_mode,
            sr=sr,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        x = torch.from_numpy(x).float()
        yid = torch.tensor(self.label2id[label], dtype=torch.long)

        item = (x, yid)

        if self.cache:
            self._cache[idx] = item

        return item


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(4, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class AblationCRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        fusion_channels=16,
        cnn_channels=(32, 64, 128),
        rnn_hidden=128,
        use_se=True,
        fusion_type="direct",
        aux_gate_init=-4.0,
        rnn_type="gru",
        pooling_type="mean",
    ):
        super().__init__()

        if fusion_type not in {"direct", "gated"}:
            raise ValueError(f"Unknown fusion_type={fusion_type}")

        if rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unknown rnn_type={rnn_type}")

        if pooling_type not in {"mean", "attn"}:
            raise ValueError(f"Unknown pooling_type={pooling_type}")

        self.in_channels = int(in_channels)
        self.fusion_type = fusion_type
        self.rnn_type = rnn_type
        self.pooling_type = pooling_type

        # For gated fusion:
        # channel 0 is Log-Mel and is always kept.
        # channels 1..C-1 are auxiliary channels.
        #
        # sigmoid(-4.0) ≈ 0.018, so auxiliary channels are nearly closed at init.
        if self.in_channels > 1 and self.fusion_type == "gated":
            self.aux_gate = nn.Parameter(
                torch.full(
                    (self.in_channels - 1,),
                    float(aux_gate_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.aux_gate = None

        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            SEBlock(fusion_channels, reduction=4) if use_se else IdentityBlock(),
        )

        blocks = []
        c_in = fusion_channels

        for c_out in cnn_channels:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ]
            c_in = c_out

        self.cnn = nn.Sequential(*blocks)

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cnn_channels[-1],
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=cnn_channels[-1],
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        else:
            raise ValueError(f"Unknown rnn_type={self.rnn_type}")

        # Temporal attention pooling:
        # mean pooling treats all time steps equally;
        # attention pooling learns which time frames are more discriminative.
        if self.pooling_type == "attn":
            self.attn = nn.Sequential(
                nn.Linear(rnn_hidden * 2, rnn_hidden),
                nn.Tanh(),
                nn.Linear(rnn_hidden, 1),
            )
        else:
            self.attn = None

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden * 2, num_classes),
        )

    def apply_input_gate(self, x):
        """
        x: [B, C, F, T]

        For gated fusion:
        - channel 0 = Log-Mel, always unchanged
        - channel 1..C-1 = auxiliary channels, scaled by learnable gates
        """
        if self.aux_gate is None:
            return x

        if x.size(1) <= 1:
            return x

        logmel = x[:, :1, :, :]
        aux = x[:, 1:, :, :]

        gate = torch.sigmoid(self.aux_gate).view(1, -1, 1, 1)
        aux = aux * gate

        return torch.cat([logmel, aux], dim=1)

    def forward(self, x):
        # x: [B, C, n_mels, T]
        x = self.apply_input_gate(x)

        x = self.frontend(x)
        x = self.cnn(x)          # [B, C, F, T]
        x = x.mean(dim=2)        # [B, C, T]
        x = x.permute(0, 2, 1)   # [B, T, C]

        out, _ = self.rnn(x)     # [B, T, 2*rnn_hidden]

        if self.pooling_type == "mean":
            z = out.mean(dim=1)

        elif self.pooling_type == "attn":
            score = self.attn(out).squeeze(-1)      # [B, T]
            weight = torch.softmax(score, dim=1)    # [B, T]
            z = torch.sum(out * weight.unsqueeze(-1), dim=1)

        else:
            raise ValueError(f"Unknown pooling_type={self.pooling_type}")

        logits = self.classifier(z)

        return logits

    def get_aux_gate_values(self):
        if self.aux_gate is None:
            return None
        return torch.sigmoid(self.aux_gate.detach().cpu()).tolist()


def compute_class_weights(ds, labels):
    counts = np.zeros(len(labels), dtype=np.float32)

    for _, row in ds.df.iterrows():
        lab = str(row[ds.label_col]).strip().lower()
        counts[ds.label2id[lab]] += 1

    weights = counts.sum() / (len(labels) * np.maximum(counts, 1.0))

    return torch.tensor(weights, dtype=torch.float32), counts


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()

    y_true = []
    y_pred = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()

        y_pred.extend(pred.tolist())
        y_true.extend(yb.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    return acc, mf1, cm, y_true, y_pred


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--fmin", type=float, default=50)
    ap.add_argument("--fmax", type=float, default=8000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=320)
    ap.add_argument("--win_length", type=int, default=800)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--feature_mode", required=True, choices=[
    "logmel",
    "logmel_hpf",
    "logmel_sg",
    "logmel_pcen",
    "logmel_sg_pcen",
    "logmel_dual",
    "logmel_dual_pcen",

    "mel_mfcc",
    "mel_mfcc_dyn",
    "mel_mfcc_dyn_flux",
    "mctafd",
],)
    ap.add_argument("--no_se", action="store_true")

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--dur_s", type=float, default=1.0)

    # early stopping / scheduler
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min_delta", type=float, default=5e-4)
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument("--boundary_loss_weight", type=float, default=0.0)
    ap.add_argument("--boundary_warmup_epochs", type=int, default=0)

    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--specaug_freq_masks", type=int, default=2)
    ap.add_argument("--specaug_time_masks", type=int, default=2)
    ap.add_argument("--specaug_freq_width", type=int, default=8)
    ap.add_argument("--specaug_time_width", type=int, default=12)

    ap.add_argument(
        "--fusion_type",
        type=str,
        default="direct",
        choices=["direct", "gated"],
    )

    ap.add_argument("--aux_gate_init", type=float, default=-4.0)

    ap.add_argument(
        "--rnn_type",
        type=str,
        default="gru",
        choices=["gru", "lstm"],
    )

    ap.add_argument(
        "--pooling_type",
        type=str,
        default="mean",
        choices=["mean", "attn"],
    )

    args = ap.parse_args()

    set_seed(args.seed)

    labels = [x.strip().lower() for x in args.labels.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] device =", device)
    print("[INFO] labels =", labels)
    print("[INFO] feature_mode =", args.feature_mode)
    print("[INFO] use_se =", not args.no_se)

    ds_train = PigVocalFeatureDataset(
        args.train,
        labels,
        feature_mode=args.feature_mode,
        sr=args.sr,
        dur_s=args.dur_s,
        cache=True,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    ds_val = PigVocalFeatureDataset(
        args.val,
        labels,
        feature_mode=args.feature_mode,
        sr=args.sr,
        dur_s=args.dur_s,
        cache=True,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    ds_test = PigVocalFeatureDataset(
        args.test,
        labels,
        feature_mode=args.feature_mode,
        sr=args.sr,
        dur_s=args.dur_s,
        cache=True,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    print("[INFO] train =", len(ds_train))
    print("[INFO] val   =", len(ds_val))
    print("[INFO] test  =", len(ds_test))

    weights, counts = compute_class_weights(ds_train, labels)

    print("[INFO] train label counts =", dict(zip(labels, counts.astype(int).tolist())))
    print("[INFO] class weights =", weights.tolist())

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    in_ch = feature_channels(args.feature_mode)

    model = AblationCRNN(
        num_classes=len(labels),
        in_channels=in_ch,
        use_se=not args.no_se,
        fusion_type=args.fusion_type,
        aux_gate_init=args.aux_gate_init,
        rnn_type=args.rnn_type,
        pooling_type=args.pooling_type,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val = -1.0
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            if args.specaug:
                xb = apply_specaugment(
                    xb,
                    freq_masks=args.specaug_freq_masks,
                    time_masks=args.specaug_time_masks,
                    freq_width=args.specaug_freq_width,
                    time_width=args.specaug_time_width,
                )

            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            if args.boundary_loss_weight > 0:
                feeding_idx = labels.index("feeding")
                stress_idx = labels.index("stress_vocal")

                mask = (yb == feeding_idx) | (yb == stress_idx)

                if mask.any():
                    fs_logits = logits[mask][:, [feeding_idx, stress_idx]]
                    fs_target = (yb[mask] == stress_idx).long()

                    fs_loss = torch.nn.functional.cross_entropy(fs_logits, fs_target)

                    if args.boundary_warmup_epochs > 0:
                        warmup_scale = min(1.0, epoch / float(args.boundary_warmup_epochs))
                    else:
                        warmup_scale = 1.0

                    loss = loss + args.boundary_loss_weight * warmup_scale * fs_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            losses.append(float(loss.item()))

        val_acc, val_mf1, val_cm, _, _ = evaluate(model, dl_val, device, labels)

        scheduler.step(val_mf1)
        current_lr = optim.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_mf1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        improved = val_mf1 > best_val + args.min_delta

        if improved:
            best_val = val_mf1
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": labels,
                    "args": vars(args),
                    "best_val_macro_f1": best_val,
                    "best_epoch": best_epoch,
                },
                args.ckpt,
            )

            print(f"[+] saved best -> {args.ckpt}  best_val={best_val:.4f} epoch={best_epoch}")

        else:
            bad_epochs += 1
            print(f"[EARLYSTOP] no improvement: {bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(
                f"[EARLYSTOP] stopped at epoch={epoch}, "
                f"best_epoch={best_epoch}, best_val={best_val:.4f}"
            )
            break

    print("[INFO] loading best checkpoint")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_acc, test_mf1, test_cm, y_true, y_pred = evaluate(
        model,
        dl_test,
        device,
        labels,
    )

    print("\n[TEST]")
    print(f"ACC={test_acc:.4f}")
    print(f"macro_f1={test_mf1:.4f}")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            digits=4,
        )
    )

    print("[CONFUSION]")
    print(test_cm)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": [labels[i] for i in y_true],
        "pred_label": [labels[i] for i in y_pred],
    })

    pred_df.to_csv(out_dir / "test_pred.csv", index=False, encoding="utf-8-sig")

    summary = {
        "feature_mode": args.feature_mode,
        "use_se": not args.no_se,
        "seed": args.seed,

        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "sr": args.sr,
        "dur_s": args.dur_s,

        "boundary_loss_weight": args.boundary_loss_weight,
        "boundary_warmup_epochs": args.boundary_warmup_epochs,
        "fusion_type": args.fusion_type,
        "aux_gate_init": args.aux_gate_init,
        "rnn_type": args.rnn_type,

        "best_val_macro_f1": float(best_val),
        "best_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_mf1),
        "confusion": test_cm.tolist(),
        "labels": labels,

        "specaug": bool(args.specaug),
        "specaug_freq_masks": args.specaug_freq_masks,
        "specaug_time_masks": args.specaug_time_masks,
        "specaug_freq_width": args.specaug_freq_width,
        "specaug_time_width": args.specaug_time_width,

        "pooling_type": args.pooling_type,
    }

    if hasattr(model, "get_aux_gate_values"):
        summary["aux_gate"] = model.get_aux_gate_values()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] wrote ->", out_dir / "test_pred.csv")
    print("[OK] wrote ->", out_dir / "summary.json")


if __name__ == "__main__":
    main()