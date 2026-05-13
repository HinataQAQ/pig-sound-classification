import argparse
import os
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing.denoise import spectral_subtraction
from src.preprocessing.vad import detect_activity_regions
from src.utils.audio_io import fix_length, load_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--do_denoise", action="store_true")
    args = ap.parse_args()

    y, _ = load_wav(args.wav, sr=args.sr, mono=True)
    target_len = int(round(args.seconds * args.sr))
    if len(y) < target_len:
        y = fix_length(y, target_len)
    else:
        y = y[:target_len]

    if args.do_denoise:
        y = spectral_subtraction(y, args.sr)

    vad_info = detect_activity_regions(y, sr=args.sr)
    times_s = vad_info["times_s"]
    score = vad_info["score"]
    thr = float(vad_info["threshold"])
    peak_idx = vad_info["peak_indices"]
    regions = vad_info["regions"]

    hop_length = int(round(0.010 * args.sr))
    win_length = int(round(0.025 * args.sr))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=args.sr,
        n_fft=1024,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=64,
        fmin=50,
        fmax=8000,
        power=2.0,
        center=False,
    )
    logmel = librosa.power_to_db(mel, ref=1.0)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    wav_t = np.arange(len(y)) / float(args.sr)
    ax1.plot(wav_t, y, linewidth=0.8)
    for r in regions:
        ax1.axvspan(r["start_s"], r["end_s"], alpha=0.15)
    ax1.set_title("Waveform + VAD candidate regions")
    ax1.set_xlabel("Time (s)")

    ax2 = fig.add_subplot(3, 1, 2)
    img = librosa.display.specshow(
        logmel,
        sr=args.sr,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
        hop_length=hop_length,
        ax=ax2,
    )
    for r in regions:
        ax2.axvline(r["peak_time_s"], linestyle="--", linewidth=1.0)
    ax2.set_title("Log-Mel spectrogram (mentor version for visual check)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(times_s, score, label="activity score")
    ax3.axhline(thr, linestyle="--", label=f"threshold={thr:.2f}")
    if len(peak_idx) > 0:
        ax3.scatter(times_s[peak_idx], score[peak_idx], s=25, label="peaks")
    ax3.set_title("Band-energy / flux VAD score")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right")

    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.wav).with_suffix(".logmel_peaks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("[OK] saved ->", out)


if __name__ == "__main__":
    main()
