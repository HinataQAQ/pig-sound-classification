import numpy as np
import librosa

def compute_mfcc(y, sr=16000, n_mfcc=26, n_mels=64, fmin=50, fmax=8000, hop_length=None, win_length=None):
    if hop_length is None:
        hop_length = int(0.010 * sr)
    if win_length is None:
        win_length = int(0.025 * sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length, win_length=win_length,
                                         n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc)
    return mfcc.T

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
