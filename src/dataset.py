import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils.audio_io import load_wav, fix_length
from .preprocessing.denoise import spectral_subtraction
from .features.mfcc import compute_mfcc, add_deltas as mfcc_delta
from .features.gfcc import compute_gfcc, add_deltas as gfcc_delta

class PigAudioDataset(Dataset):
    def __init__(self, manifest_path, labels, sr=16000, segment_seconds=2.0,
                 do_denoise=True, do_vad=False, include_delta=True, include_delta_delta=False,
                 train=True):
        self.df = pd.read_csv(manifest_path)

        # 规范化配置里的标签（小写 + 去首尾空格），只建一次映射
        self.labels = [str(l).strip().lower() for l in labels]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        self.sr = sr
        self.seg_len = int(sr * segment_seconds)
        self.do_denoise = do_denoise
        self.do_vad = do_vad
        self.include_delta = include_delta
        self.include_delta_delta = include_delta_delta
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 统一时间参数：25ms 窗 + 10ms 步
        hop_len = int(0.010 * self.sr)   # 10 ms
        win_len = int(0.025 * self.sr)   # 25 ms
        hop_time = hop_len / float(self.sr)
        window_time = win_len / float(self.sr)

        row = self.df.iloc[idx]
        path = row["filepath"]

        # 标签规范化后再查表
        raw_lab = str(row["label"])
        lab_norm = raw_lab.strip().lower()
        if lab_norm not in self.label2id:
            raise KeyError(
                f"Label '{raw_lab}' (normalized '{lab_norm}') not in config labels {self.labels}. "
                f"Fix config.data.labels to match manifests."
            )
        label = self.label2id[lab_norm]

        # 读取与预处理
        y, sr = load_wav(path, sr=self.sr, mono=True)
        if self.do_denoise:
            y = spectral_subtraction(y, sr)

        # 取 2 秒片段（训练取随机窗口，验证取中间窗口）
        if len(y) < self.seg_len:
            y = fix_length(y, self.seg_len)
        else:
            start = random.randint(0, len(y) - self.seg_len) if self.train else max(0, (len(y) - self.seg_len)//2)
            y = y[start:start + self.seg_len]

        # 1) MFCC（与 GFCC 时间步一致，且不做中心 padding）
        mfcc = compute_mfcc(y, sr=self.sr, hop_length=hop_len, win_length=win_len, center=False)
        mfcc = mfcc_delta(mfcc, self.include_delta, self.include_delta_delta)

        # 2) GFCC（与 MFCC 时间等效）
        gfcc = compute_gfcc(y, sr=self.sr, n_gfcc=32, n_filters=64,
                            window_time=window_time, hop_time=hop_time, fmin=50, fmax=8000)
        gfcc = gfcc_delta(gfcc, self.include_delta, self.include_delta_delta)

        # 3) 先对齐帧数，再拼接（这是关键）
        T = min(mfcc.shape[0], gfcc.shape[0])
        mfcc = mfcc[:T]
        gfcc = gfcc[:T]
        feat = np.concatenate([mfcc, gfcc], axis=1).astype(np.float32)

        # Utterance-level 均值方差归一化
        mu = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / std

        x = torch.from_numpy(feat)   # (T, F)
        ylab = torch.tensor(label).long()
        return x, ylab
