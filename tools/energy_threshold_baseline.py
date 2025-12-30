import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

def load_mono(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return y

def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y * y) + 1e-12))

def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels=("cough", "other")) -> float:
    f1s = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        fp = np.sum((y_true != lab) & (y_pred == lab))
        fn = np.sum((y_true == lab) & (y_pred != lab))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def eval_with_thr(df: pd.DataFrame, thr: float, cough_if_high: bool):
    # cough_if_high=True  => rms > thr 预测 cough
    # cough_if_high=False => rms < thr 预测 cough（方向相反也试一下）
    if cough_if_high:
        pred = np.where(df["rms"].values > thr, "cough", "other")
    else:
        pred = np.where(df["rms"].values < thr, "cough", "other")

    y_true = df["label"].values
    acc = float(np.mean(pred == y_true))
    mf1 = f1_macro(y_true, pred)
    return acc, mf1, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    args = ap.parse_args()

    def load_df(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["filepath"] = df["filepath"].astype(str)
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        return df

    dfv = load_df(args.val_csv)
    dft = load_df(args.test_csv)

    # 计算 RMS
    for df in (dfv, dft):
        rms_list = []
        for p in df["filepath"].values:
            y = load_mono(p)
            rms_list.append(rms_energy(y))
        df["rms"] = rms_list

    # 在 val 上选阈值（同时试两种方向）
    cand = np.unique(dfv["rms"].values)
    best = (-1.0, -1.0, None, None)  # (acc, mf1, thr, cough_if_high)
    for cough_if_high in (True, False):
        for thr in cand:
            acc, mf1, _ = eval_with_thr(dfv, thr, cough_if_high)
            if mf1 > best[1]:
                best = (acc, mf1, float(thr), cough_if_high)

    best_acc, best_mf1, best_thr, best_dir = best
    dir_str = "rms>thr => cough" if best_dir else "rms<thr => cough"
    print(f"=== VAL best by Macro-F1 ===")
    print(f"best_thr={best_thr:.6f}  dir=({dir_str})  ACC={best_acc:.4f}  Macro-F1={best_mf1:.4f}")

    # 用 best_thr 在 test 上评估
    acc_t, mf1_t, pred_t = eval_with_thr(dft, best_thr, best_dir)
    print(f"\n=== TEST (using VAL best_thr) ===")
    print(f"ACC={acc_t:.4f}  Macro-F1={mf1_t:.4f}")

    # 混淆矩阵（cough / other）
    y_true = dft["label"].values
    cm = np.zeros((2, 2), dtype=int)
    idx = {"cough": 0, "other": 1}
    for yt, yp in zip(y_true, pred_t):
        cm[idx[yt], idx[yp]] += 1
    print("\nconfusion [[true_cough->pred_cough, true_cough->pred_other], [true_other->pred_cough, true_other->pred_other]]:")
    print(cm)

if __name__ == "__main__":
    main()
