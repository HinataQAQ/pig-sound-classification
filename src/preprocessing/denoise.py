import numpy as np
import librosa

def spectral_subtraction(y, sr, n_fft=1024, hop_length=256, win_length=1024, pre_noise_sec=0.2, alpha=1.0):
    """
    Simple spectral subtraction (Boll 1979). Assume first pre_noise_sec contains mostly noise.
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
    mag, phase = np.abs(S), np.angle(S)
    n_frames = int(pre_noise_sec * sr / hop_length)
    n_frames = max(1, min(mag.shape[1]//4, n_frames))
    noise_mag = np.mean(mag[:, :n_frames], axis=1, keepdims=True)
    clean_mag = np.maximum(mag - alpha * noise_mag, 0.0)
    S_clean = clean_mag * np.exp(1j * phase)
    y_hat = librosa.istft(S_clean, hop_length=hop_length, win_length=win_length, window="hann")
    return y_hat.astype(np.float32)
