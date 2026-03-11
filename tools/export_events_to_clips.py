# tools/export_events_to_clips.py
import os
import re
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

# Windows 下常见 OMP 冲突的临时绕过（可选，但你之前确实遇到过）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="path to long wav")
    ap.add_argument("--events", required=True, help="events csv from detect_long_cough.py")
    ap.add_argument("--out_dir", default="det_clips", help="output folder for wav clips")
    ap.add_argument("--pad", type=float, default=0.0, help="pad seconds before/after each event")
    ap.add_argument("--min_dur", type=float, default=0.0, help="skip events shorter than this (seconds)")
    ap.add_argument("--max_dur", type=float, default=0.0, help="cap duration to this (seconds), 0 disables")
    ap.add_argument("--prefix", type=str, default="", help="optional filename prefix")
    args = ap.parse_args()

    df = pd.read_csv(args.events)

    # 兼容你的 events.csv 列名：start_s/end_s
    start_col = _find_col(df, ["start_sec", "start_s", "start", "onset", "begin_s", "begin_sec"])
    end_col = _find_col(df, ["end_sec", "end_s", "end", "offset", "finish_s", "finish_sec"])

    if start_col is None or end_col is None:
        raise KeyError(
            f"Cannot find start/end columns in events csv. columns={list(df.columns)}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    exported = 0
    wav_abs = os.path.abspath(args.wav)
    events_abs = os.path.abspath(args.events)

    # 用 SoundFile 流式读，不把整段长音频一次性读进内存（更稳）
    with sf.SoundFile(args.wav, "r") as f:
        sr = int(f.samplerate)
        total_frames = int(len(f))

        for i, r in df.iterrows():
            s = float(r[start_col])
            e = float(r[end_col])

            if (not np.isfinite(s)) or (not np.isfinite(e)) or (e <= s):
                continue

            # pad
            s = max(0.0, s - float(args.pad))
            e = e + float(args.pad)

            # cap duration
            if args.max_dur and args.max_dur > 0:
                e = min(e, s + float(args.max_dur))

            dur = e - s
            if dur < float(args.min_dur):
                continue

            s_idx = int(round(s * sr))
            e_idx = int(round(e * sr))

            # clamp
            s_idx = max(0, min(s_idx, total_frames))
            e_idx = max(0, min(e_idx, total_frames))
            if e_idx <= s_idx:
                continue

            f.seek(s_idx)
            clip = f.read(e_idx - s_idx, dtype="float32", always_2d=False)

            # to mono if needed
            if isinstance(clip, np.ndarray) and clip.ndim == 2:
                clip = clip.mean(axis=1).astype(np.float32)

            # 文件名里带上 max_prob（如果存在）
            extra = ""
            if "max_prob" in df.columns:
                try:
                    extra += f"_p{float(r['max_prob']):.3f}"
                except Exception:
                    pass

            name_prefix = _safe(args.prefix) + "_" if args.prefix else ""
            out_name = f"{name_prefix}event_{i:04d}_{s:.2f}-{e:.2f}{extra}.wav"
            out_path = os.path.join(args.out_dir, out_name)

            sf.write(out_path, clip, sr)
            exported += 1

    print("[OK] wav     =", wav_abs)
    print("[OK] events  =", events_abs)
    print("[OK] out_dir =", os.path.abspath(args.out_dir))
    print(f"[OK] exported clips = {exported}")


if __name__ == "__main__":
    main()
