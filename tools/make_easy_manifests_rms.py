import argparse, glob, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split


def wav_rms(path: str) -> float:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return float(np.sqrt(np.mean(y * y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/processed/cough_silence_binary",
                    help="folder contains cough/*.wav and other/*.wav")
    ap.add_argument("--out_dir", default="data/manifests_cough_silence_easy")
    ap.add_argument("--cough_keep_top", type=float, default=0.25,
                    help="keep top X of cough by RMS (e.g. 0.25 => top 25%)")
    ap.add_argument("--other_keep_bottom", type=float, default=0.25,
                    help="keep bottom X of other by RMS (e.g. 0.25 => bottom 25%)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cough_files = sorted(glob.glob(os.path.join(args.in_root, "cough", "*.wav")))
    other_files = sorted(glob.glob(os.path.join(args.in_root, "other", "*.wav")))
    assert len(cough_files) > 0 and len(other_files) > 0, f"No wav files under {args.in_root}"

    cough_r = np.array([wav_rms(p) for p in cough_files])
    other_r = np.array([wav_rms(p) for p in other_files])

    cough_thr = float(np.quantile(cough_r, 1.0 - args.cough_keep_top))
    other_thr = float(np.quantile(other_r, args.other_keep_bottom))

    cough_sel = [p for p, r in zip(cough_files, cough_r) if r >= cough_thr]
    other_sel = [p for p, r in zip(other_files, other_r) if r <= other_thr]

    # 对齐数量（否则训练会偏）
    n = min(len(cough_sel), len(other_sel))
    random.shuffle(cough_sel)
    random.shuffle(other_sel)
    cough_sel = cough_sel[:n]
    other_sel = other_sel[:n]

    rows = (
        [{"filepath": p.replace("\\", "/"), "label": "cough"} for p in cough_sel]
        + [{"filepath": p.replace("\\", "/"), "label": "other"} for p in other_sel]
    )
    random.shuffle(rows)
    df = pd.DataFrame(rows)

    tmp_ratio = args.val_ratio + args.test_ratio
    df_train, df_tmp = train_test_split(
        df, test_size=tmp_ratio, stratify=df["label"], random_state=args.seed
    )
    rel_test = args.test_ratio / tmp_ratio
    df_val, df_test = train_test_split(
        df_tmp, test_size=rel_test, stratify=df_tmp["label"], random_state=args.seed
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out / "train.csv", index=False)
    df_val.to_csv(out / "val.csv", index=False)
    df_test.to_csv(out / "test.csv", index=False)

    print(f"[OK] in_root={args.in_root}")
    print(f"[OK] cough_thr(q={1.0-args.cough_keep_top:.2f}) = {cough_thr:.6f}")
    print(f"[OK] other_thr(q={args.other_keep_bottom:.2f}) = {other_thr:.6f}")
    print(f"[OK] selected per-class n={n}  total={len(df)}")
    print("train:\n", df_train["label"].value_counts())
    print("val:\n", df_val["label"].value_counts())
    print("test:\n", df_test["label"].value_counts())


if __name__ == "__main__":
    main()
