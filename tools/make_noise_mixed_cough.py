from pathlib import Path
import argparse
import random
import math
import pandas as pd
import numpy as np
import librosa
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def collect_audio(dirs):
    files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] missing dir: {p}")
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                files.append(f)
    return sorted(files)


def load_fixed(path: Path, sr: int, dur_s: float, rng: random.Random):
    target_len = int(sr * dur_s)
    y, _ = librosa.load(str(path), sr=sr, mono=True)

    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    y = y.astype(np.float32)

    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        max_start = len(y) - target_len
        start = rng.randint(0, max_start)
        y = y[start:start + target_len]

    return y


def rms_power(x):
    return float(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def mix_with_snr(clean, noise, snr_db):
    clean_power = rms_power(clean)
    noise_power = rms_power(noise)

    # desired: SNR = clean_power / scaled_noise_power
    desired_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(desired_noise_power / noise_power)

    mixed = clean + noise * scale

    peak = float(np.max(np.abs(mixed)) + 1e-9)
    if peak > 0.98:
        mixed = mixed / peak * 0.98

    return mixed.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cough_dirs", nargs="+", required=True)
    ap.add_argument("--noise_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_manifest", required=True)

    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--dur_s", type=float, default=1.0)
    ap.add_argument("--snrs", nargs="+", type=float, default=[0.0, 5.0, 10.0])
    ap.add_argument("--seed", type=int, default=3407)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    cough_files = collect_audio(args.cough_dirs)
    noise_files = collect_audio(args.noise_dirs)

    if len(cough_files) == 0:
        raise RuntimeError("No cough files found.")
    if len(noise_files) == 0:
        raise RuntimeError("No noise files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] cough files = {len(cough_files)}")
    print(f"[INFO] noise files = {len(noise_files)}")
    print(f"[INFO] output n = {args.n}")

    rows = []
    made = 0
    tries = 0
    max_tries = args.n * 20

    while made < args.n and tries < max_tries:
        tries += 1

        cough_path = rng.choice(cough_files)
        noise_path = rng.choice(noise_files)
        snr = rng.choice(args.snrs)

        try:
            clean = load_fixed(cough_path, args.sr, args.dur_s, rng)
            noise = load_fixed(noise_path, args.sr, args.dur_s, rng)
            mixed = mix_with_snr(clean, noise, snr)

            out_name = f"mix_{made:05d}__snr{snr:g}__cough_{cough_path.stem}__noise_{noise_path.stem}.wav"
            # 文件名太长时 Windows 可能麻烦，所以再截短
            if len(out_name) > 180:
                out_name = f"mix_{made:05d}__snr{snr:g}.wav"

            out_path = out_dir / out_name
            sf.write(str(out_path), mixed, args.sr)

            rows.append({
                "filepath": str(out_path.resolve()),
                "label": "cough",
                "source": "noise_mixed_cough",
                "cough_source": str(cough_path.resolve()),
                "noise_source": str(noise_path.resolve()),
                "snr_db": snr,
            })

            made += 1

        except Exception as e:
            print(f"[WARN] skip pair cough={cough_path} noise={noise_path} err={repr(e)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_manifest, index=False, encoding="utf-8-sig")

    print(f"[DONE] generated={len(df)}")
    print(f"[DONE] manifest={args.out_manifest}")


if __name__ == "__main__":
    main()