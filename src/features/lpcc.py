import numpy as np
import librosa

def compute_lpcc(y, sr=16000, order=16, n_lpcc=20, hop_length=None, win_length=None):
    if hop_length is None:
        hop_length = int(0.010 * sr)
    if win_length is None:
        win_length = int(0.025 * sr)
    frames = librosa.util.frame(y, frame_length=win_length, hop_length=hop_length).T
    frames = frames * np.hanning(win_length)[None, :]
    lpcc_all = []
    for frame in frames:
        a = librosa.lpc(frame, order=order)
        lpcc_all.append(_lpc_to_lpcc(a, n_lpcc))
    return np.stack(lpcc_all, axis=0)

def _lpc_to_lpcc(a, n_lpcc):
    p = len(a)-1
    c = np.zeros(n_lpcc, dtype=np.float32)
    c[0] = 0.0
    for n in range(1, n_lpcc):
        s = 0.0
        kmax = min(n, p)
        for k in range(1, kmax):
            if (n-k) <= p:
                s += (k/n) * c[k] * a[n-k]
        if n <= p:
            c[n] = -a[n] + s
        else:
            c[n] = s
    return c
