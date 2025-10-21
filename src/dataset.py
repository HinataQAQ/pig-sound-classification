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
        self.labels = labels
        self.label2id = {l:i for i,l in enumerate(labels)}
        self.id2label = {i:l for l,i in self.label2id.items()}
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
        row = self.df.iloc[idx]
        path = row["filepath"]
        label = self.label2id[row["label"]]
        y, sr = load_wav(path, sr=self.sr, mono=True)
        if self.do_denoise:
            y = spectral_subtraction(y, sr)
        if len(y) < self.seg_len:
            y = fix_length(y, self.seg_len)
        else:
            start = random.randint(0, len(y) - self.seg_len) if self.train else max(0, (len(y) - self.seg_len)//2)
            y = y[start:start+self.seg_len]

        mfcc = compute_mfcc(y, sr=self.sr)
        mfcc = mfcc_delta(mfcc, self.include_delta, self.include_delta_delta)
        gfcc = compute_gfcc(y, sr=self.sr)
        gfcc = gfcc_delta(gfcc, self.include_delta, self.include_delta_delta)
        feat = np.concatenate([mfcc, gfcc], axis=1).astype(np.float32)

        mu = feat.mean(axis=0, keepdims=True); std = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / std

        x = torch.from_numpy(feat)         # (T, F)
        ylab = torch.tensor(label).long()
        return x, ylab
