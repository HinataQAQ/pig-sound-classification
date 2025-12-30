import argparse
import numpy as np
import librosa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--fmin", type=int, default=50)
    ap.add_argument("--fmax", type=int, default=8000)
    args = ap.parse_args()

    y, sr = librosa.load(args.wav, sr=args.sr, mono=True)
    target_len = int(args.seconds * args.sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # 25ms window, 10ms hop (常见语音/音频设置)
    win_length = int(0.025 * args.sr)
    hop_length = int(0.010 * args.sr)
    n_fft = 1024  # >= win_length, 且是2的幂

    mel = librosa.feature.melspectrogram(
        y=y, sr=args.sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        power=2.0,
        center=False
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # 训练时我们更喜欢 (T, F)
    feat = logmel.T

    print("[OK] wav =", args.wav)
    print("[OK] sr =", sr, "len =", len(y), "seconds =", len(y)/sr)
    print("[OK] mel shape (F, T) =", mel.shape)
    print("[OK] logmel shape (T, F) =", feat.shape)
    print("[OK] logmel stats: min=%.3f max=%.3f mean=%.3f" % (feat.min(), feat.max(), feat.mean()))

if __name__ == "__main__":
    main()
