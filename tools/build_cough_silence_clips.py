# tools/build_cough_silence_clips.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

def resolve_aswine_audio_path(audio_file_path: str, meta_csv: Path, aswine_root: Path | None):
    # meta_csv: .../data/external/aswine/meta/1s_pruned/xxx.csv
    # aswine_root 默认推断为 .../data/external/aswine
    if aswine_root is None:
        aswine_root = meta_csv.resolve().parents[2]

    rel = str(audio_file_path).strip().replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    rel_path = Path(rel)

    cand1 = (aswine_root / rel_path).resolve()
    if cand1.exists():
        return cand1

    # 兜底：如果有人把 csv 移走了，尝试相对 csv 所在目录
    cand2 = (meta_csv.parent / rel_path).resolve()
    if cand2.exists():
        return cand2

    raise FileNotFoundError(
        f"Cannot find wav for audio_file_path='{audio_file_path}'.\n"
        f"Tried:\n - {cand1}\n - {cand2}\n"
        f"Hint: pass --aswine_root pointing to dataset root (the folder that contains 'data/audio')."
    )

def read_segment(wav_path: Path, offset_s: float, duration_s: float):
    with sf.SoundFile(str(wav_path)) as f:
        sr = f.samplerate
        start = int(round(offset_s * sr))
        frames = int(round(duration_s * sr))
        if start < 0:
            start = 0
        f.seek(start)
        y = f.read(frames, dtype="float32", always_2d=True)  # (T, C)
    # 转 mono
    y = y.mean(axis=1)
    # 不足补零
    if len(y) < frames:
        y = np.pad(y, (0, frames - len(y)), mode="constant")
    else:
        y = y[:frames]
    return y, sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, default=r"data\external\aswine\meta\1s_pruned\aswine_pure_cough_silence_other.csv")
    ap.add_argument("--aswine_root", type=str, default="", help="optional: path to aswine dataset root")
    ap.add_argument("--out_root", type=str, default=r"data\processed\cough_silence_binary")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 means full; >0 means only first N rows")
    args = ap.parse_args()

    meta_csv = Path(args.meta_csv)
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta_csv not found: {meta_csv.resolve()}")

    aswine_root = Path(args.aswine_root) if args.aswine_root.strip() else None
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)
    df = df.sample(frac=1, random_state=3407).reset_index(drop=True)
    need_cols = ["audio_file_path", "offset", "duration", "label"]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {meta_csv}. Got: {list(df.columns)}")

    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    saved = {"cough": 0, "other": 0}
    first_out = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="cut clips"):
        lab = str(row["label"]).strip().lower()
        if lab not in ("cough", "other"):
            continue

        src = resolve_aswine_audio_path(row["audio_file_path"], meta_csv, aswine_root)
        offset = float(row["offset"])
        duration = float(row["duration"])

        # 输出路径
        out_dir = out_root / lab
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = src.stem.replace(" ", "_")
        t_ms = int(round(offset * 1000.0))
        out_name = f"aswine_{stem}_t{t_ms:010d}.wav"
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            continue

        y, sr = read_segment(src, offset, duration)
        sf.write(str(out_path), y, sr, subtype="PCM_16")

        saved[lab] += 1
        if first_out is None:
            first_out = str(out_path)

    print(f"\n[DONE] out_root = {out_root}")
    print(f"[DONE] saved cough = {saved['cough']}")
    print(f"[DONE] saved other = {saved['other']}")
    if first_out:
        print(f"[INFO] first file = {first_out}")

if __name__ == "__main__":
    main()
