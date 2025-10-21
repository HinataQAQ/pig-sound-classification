import os
import numpy as np
import librosa
import soundfile as sf

def load_wav(path, sr=16000, mono=True):
    """Load wav with librosa, return float32 mono array and sr."""
    y, s = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), s

def save_wav(path, y, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)

def fix_length(y, target_len):
    if len(y) >= target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")
