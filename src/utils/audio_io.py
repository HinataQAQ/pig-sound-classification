from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav(path, sr=None, mono=True):
    """
    Stable audio loader for this project.

    使用 soundfile 直接读取 wav/flac/ogg 等音频，避免部分文件在 librosa.load 阶段卡住。
    soundfile 基于 libsndfile，官方文档说明它可读写多种声音文件格式。
    """
    path = str(Path(path))

    y, s = sf.read(path, dtype="float32", always_2d=False)

    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    # stereo / multi-channel -> mono
    if isinstance(y, np.ndarray) and y.ndim == 2:
        if mono:
            y = y.mean(axis=1)
        else:
            y = y.T

    y = np.asarray(y, dtype=np.float32)

    # resample if needed
    if sr is not None and int(s) != int(sr):
        g = gcd(int(s), int(sr))
        up = int(sr) // g
        down = int(s) // g
        y = resample_poly(y, up, down).astype(np.float32)
        s = int(sr)

    return y, int(s)


def fix_length(y, target_len=None, length=None):
    """
    Pad or trim waveform to fixed length.

    兼容几种可能调用方式：
    fix_length(y, 32000)
    fix_length(y, target_len=32000)
    fix_length(y, length=32000)
    """
    if target_len is None:
        target_len = length

    if target_len is None:
        raise ValueError("target_len or length must be provided")

    target_len = int(target_len)
    y = np.asarray(y, dtype=np.float32)

    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y.astype(np.float32)