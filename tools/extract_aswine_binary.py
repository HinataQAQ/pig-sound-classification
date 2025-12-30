# tools/extract_aswine_binary.py
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except Exception as e:
    print("[ERR] missing dependency: soundfile")
    print("      Please run: pip install soundfile")
    raise


def to_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("true", "1", "t", "yes", "y")


def build_audio_index(aswine_root: Path) -> dict:
    """
    aSwine 的 CSV 里 audio_file_path 形如 ./data/audio/xxx.wav
    但你实际解压后可能在:
      - data/external/aswine/audio/*.wav
      - data/external/aswine/data/audio/*.wav
    所以我们扫描常见目录，建立 filename -> fullpath 的索引。
    """
    candidates = [
        aswine_root / "audio",
        aswine_root / "data" / "audio",
    ]
    idx = {}
    for d in candidates:
        if d.exists():
            for p in d.rglob("*.wav"):
                idx[p.name] = p
            for p in d.rglob("*.WAV"):
                idx[p.name] = p
    if not idx:
        raise FileNotFoundError(
            f"No wav files found under {candidates}. "
            f"Please check your aSwine folder structure."
        )
    return idx


def load_wav_mono(path: Path, expected_sr: int) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != expected_sr:
        raise ValueError(f"Sample rate mismatch: {path} sr={sr}, expected={expected_sr}")
    return y, sr


def write_clip(out_path: Path, clip: np.ndarray, sr: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--aswine_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--ratio", type=float, default=1.0, help="other:cough ratio, e.g. 1.0 means 1:1")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 means no limit; otherwise limit per class")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    aswine_root = Path(args.aswine_root)
    out_root = Path(args.out_root)
    cough_dir = out_root / "cough"
    other_dir = out_root / "other"

    assert csv_path.exists(), f"CSV not found: {csv_path}"
    assert aswine_root.exists(), f"aSwine root not found: {aswine_root}"

    df = pd.read_csv(csv_path)
    if "TosseEspirro" not in df.columns:
        raise KeyError("CSV missing column: TosseEspirro")

    df["is_cough"] = df["TosseEspirro"].apply(to_bool)

    df_c = df[df["is_cough"] == True].copy()
    df_o = df[df["is_cough"] == False].copy()

    n_cough = len(df_c)
    n_other = int(round(n_cough * args.ratio))
    rng = np.random.default_rng(args.seed)

    if n_other > len(df_o):
        n_other = len(df_o)

    # 抽样 other 以保持 1:1
    other_idx = rng.choice(len(df_o), size=n_other, replace=False)
    df_o = df_o.iloc[other_idx].copy()

    # 可选：限制每类数量（冒烟测试用）
    if args.max_per_class and args.max_per_class > 0:
        if len(df_c) > args.max_per_class:
            df_c = df_c.sample(args.max_per_class, random_state=args.seed)
        if len(df_o) > args.max_per_class:
            df_o = df_o.sample(args.max_per_class, random_state=args.seed)

    # 建立音频索引
    audio_idx = build_audio_index(aswine_root)

    # 统一数据结构
    df_c["label"] = "cough"
    df_o["label"] = "other"
    df_use = pd.concat([df_c, df_o], ignore_index=True)

    # 从 audio_file_path 提取文件名
    df_use["wav_name"] = df_use["audio_file_path"].astype(str).apply(lambda s: Path(s).name)

    # 分组：同一长 wav 只加载一次
    saved = {"cough": 0, "other": 0}
    missing = 0

    for wav_name, g in df_use.groupby("wav_name"):
        if wav_name not in audio_idx:
            missing += len(g)
            continue

        wav_path = audio_idx[wav_name]
        y, sr = load_wav_mono(wav_path, args.sr)

        for _, row in g.iterrows():
            label = row["label"]
            offset = float(row["offset"])
            dur = float(row["duration"])
            start = int(round(offset * sr))
            end = int(round((offset + dur) * sr))
            if end > len(y):
                clip = np.zeros(int(round(dur * sr)), dtype=np.float32)
                part = y[start:len(y)]
                clip[: len(part)] = part
            else:
                clip = y[start:end].astype(np.float32)

            stem = Path(wav_name).stem
            t_ms = int(round(offset * 1000))
            out_name = f"aswine_{stem}_t{t_ms:010d}.wav"
            out_path = (cough_dir if label == "cough" else other_dir) / out_name
            write_clip(out_path, clip, sr)
            saved[label] += 1

    print("[DONE] out_root =", out_root)
    print("[DONE] saved cough =", saved["cough"])
    print("[DONE] saved other =", saved["other"])
    if missing:
        print("[WARN] missing wav files for rows =", missing)
        print("       If missing > 0, tell me and I'll help fix path mapping.")

if __name__ == "__main__":
    main()
