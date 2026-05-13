from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
SR = 32000
DUR_S = 1.0
N_FFT = 1024
HOP_LENGTH = int(SR * 0.010)
WIN_LENGTH = int(SR * 0.025)
N_MELS = 64
N_MFCC = 20


def load_fixed_audio(path: str):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    target_len = int(SR * DUR_S)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y


def compute_display(y, feature):
    if feature == "logmel":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=50,
            fmax=8000,
            power=2.0,
        )
        return librosa.power_to_db(mel, ref=1.0, top_db=80.0), "Log-Mel"

    if feature == "pcenmel":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=50,
            fmax=8000,
            power=1.0,
        )
        return librosa.pcen(mel * (2 ** 31), sr=SR, hop_length=HOP_LENGTH), "PCEN-Mel"

    if feature == "mfcc":
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        return mfcc, "MFCC"

    if feature == "logstft":
        stft = librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        keep = (freqs >= 50) & (freqs <= 8000)
        logspec = librosa.amplitude_to_db(mag[keep, :], ref=1.0, top_db=80.0)
        return logspec, "Log-STFT"

    raise ValueError(feature)


def save_img(mat, title, out_path):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mat, x_axis="time", sr=SR, hop_length=HOP_LENGTH)
    plt.title(title)
    plt.colorbar(format="%+2.0f")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"data\manifests_source_split\train.csv")
    ap.add_argument("--out_dir", default=r"reports\feature_ablation\feature_gallery")
    ap.add_argument("--n_per_class", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.manifest)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for lab in ["cough", "other"]:
        part = df[df["label"].astype(str).str.lower() == lab].head(args.n_per_class)
        selected.append(part)

    selected_df = pd.concat(selected, ignore_index=True)

    rows = []
    for i, row in selected_df.iterrows():
        path = str(row["filepath"])
        lab = str(row["label"])
        y = load_fixed_audio(path)

        for feat in ["logmel", "pcenmel", "mfcc", "logstft"]:
            mat, name = compute_display(y, feat)
            out_name = f"{i:02d}_{lab}_{feat}.png"
            out_path = out_dir / out_name
            save_img(mat, f"{lab} - {name}", out_path)

            rows.append({
                "label": lab,
                "feature": feat,
                "source_audio": path,
                "image": str(out_path),
            })

    pd.DataFrame(rows).to_csv(out_dir / "gallery_index.csv", index=False, encoding="utf-8-sig")
    print("[OK] wrote gallery ->", out_dir)


if __name__ == "__main__":
    main()