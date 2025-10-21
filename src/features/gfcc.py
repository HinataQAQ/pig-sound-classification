import numpy as np
from gammatone.gtgram import gtgram
from scipy.fftpack import dct

def compute_gfcc(y, sr=16000, n_gfcc=32, n_filters=64, window_time=0.025, hop_time=0.010, fmin=50, fmax=8000):
    """
    GFCC via gammatone spectrogram + log + DCT
    """
    S = gtgram(y, sr, window_time, hop_time, n_filters, fmin)  # (n_filters, T)
    S = S[:n_filters, :]
    S = np.where(S <= 0, 1e-12, S)
    log_S = np.log(S)
    cep = dct(log_S, type=2, axis=0, norm="ortho")  # (n_filters, T)
    cep = cep[:n_gfcc, :].T  # (T, n_gfcc)
    return cep

def add_deltas(feat, include_delta=True, include_delta_delta=False):
    if not include_delta and not include_delta_delta:
        return feat
    import librosa
    feat_t = feat.T
    feats = [feat_t]
    if include_delta:
        feats.append(librosa.feature.delta(feat_t))
    if include_delta_delta:
        feats.append(librosa.feature.delta(feat_t, order=2))
    out = np.concatenate(feats, axis=0).T
    return out
