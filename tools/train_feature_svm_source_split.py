from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
LABELS = ["cough", "other"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}

SR = 32000
DUR_S = 1.0
N_FFT = 1024
HOP_LENGTH = int(SR * 0.010)
WIN_LENGTH = int(SR * 0.025)
N_MELS = 64
N_MFCC = 20


def load_fixed_audio(path: str, sr: int = SR, dur_s: float = DUR_S) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)

    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    target_len = int(sr * dur_s)

    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        start = max(0, (len(y) - target_len) // 2)
        y = y[start:start + target_len]

    return y.astype(np.float32)


def stat_pool(mat: np.ndarray) -> np.ndarray:
    """
    输入形状一般是 [freq, time]。
    输出固定长度：mean/std/max/min。
    """
    mat = np.asarray(mat, dtype=np.float32)

    feat = np.concatenate([
        mat.mean(axis=1),
        mat.std(axis=1),
        mat.max(axis=1),
        mat.min(axis=1),
    ], axis=0)

    return feat.astype(np.float32)


def extract_feature(path: str, feature: str) -> np.ndarray:
    y = load_fixed_audio(path)

    if feature == "mfcc":
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)

        feat = np.concatenate([
            stat_pool(mfcc),
            stat_pool(d1),
            stat_pool(d2),
        ], axis=0)
        return feat

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
        logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        return stat_pool(logmel)

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
        pcen = librosa.pcen(mel * (2 ** 31), sr=SR, hop_length=HOP_LENGTH)
        return stat_pool(pcen)

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
        mag = mag[keep, :]

        logspec = librosa.amplitude_to_db(mag, ref=1.0, top_db=80.0)

        # STFT 频点较多，为了 SVM 稳定，把频率轴粗采样到最多 128 维
        if logspec.shape[0] > 128:
            idx = np.linspace(0, logspec.shape[0] - 1, 128).astype(int)
            logspec = logspec[idx, :]

        return stat_pool(logspec)
    if feature == "fusion":
        # 1) MFCC + delta + delta2
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)

        mfcc_feat = np.concatenate([
            stat_pool(mfcc),
            stat_pool(d1),
            stat_pool(d2),
        ], axis=0)

        # 2) Log-Mel
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
        logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        logmel_feat = stat_pool(logmel)

        # 3) Log-STFT
        stft = librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        mag = np.abs(stft)

        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        keep = (freqs >= 50) & (freqs <= 8000)
        mag = mag[keep, :]

        logspec = librosa.amplitude_to_db(mag, ref=1.0, top_db=80.0)

        if logspec.shape[0] > 128:
            idx = np.linspace(0, logspec.shape[0] - 1, 128).astype(int)
            logspec = logspec[idx, :]

        logstft_feat = stat_pool(logspec)

        return np.concatenate([
            mfcc_feat,
            logmel_feat,
            logstft_feat,
        ], axis=0).astype(np.float32)
    raise ValueError(f"unknown feature: {feature}")


def load_split(csv_path: Path, feature: str, out_bad: Path):
    df = pd.read_csv(csv_path)
    X, y = [], []
    bad_rows = []

    for _, row in df.iterrows():
        p = str(row["filepath"])
        lab = str(row["label"]).strip().lower()

        try:
            X.append(extract_feature(p, feature))
            y.append(LABEL2ID[lab])
        except Exception as e:
            bad_rows.append({
                "filepath": p,
                "label": lab,
                "error": repr(e),
            })
            print(f"[BAD][{feature}] {p} -> {repr(e)}")

    pd.DataFrame(bad_rows).to_csv(out_bad, index=False, encoding="utf-8-sig")

    if len(X) == 0:
        raise RuntimeError(f"No usable samples in {csv_path}")

    return np.stack(X), np.array(y, dtype=np.int64)


def run_one(feature: str, mani_dir: Path, out_root: Path):
    out_dir = out_root / feature
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== FEATURE: {feature} ==========")

    X_train, y_train = load_split(mani_dir / "train.csv", feature, out_dir / "train_bad.csv")
    X_val, y_val = load_split(mani_dir / "val.csv", feature, out_dir / "val_bad.csv")
    X_test, y_test = load_split(mani_dir / "test.csv", feature, out_dir / "test_bad.csv")

    print("[INFO] train shape:", X_train.shape)
    print("[INFO] val shape:", X_val.shape)
    print("[INFO] test shape:", X_test.shape)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=3407)),
    ])

    clf.fit(X_train, y_train)

    results = {}
    for split_name, X, y_true in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = clf.predict(X)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print(f"\n[{feature}][{split_name}] ACC={acc:.4f} macro_f1={macro_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))
        print(cm)

        pred_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [ID2LABEL[int(v)] for v in y_true],
            "pred_label": [ID2LABEL[int(v)] for v in y_pred],
        })
        pred_df.to_csv(out_dir / f"{split_name}_pred.csv", index=False, encoding="utf-8-sig")

        results[split_name] = {
            "acc": float(acc),
            "macro_f1": float(macro_f1),
            "confusion": cm.tolist(),
        }

    joblib.dump(clf, out_dir / "model.joblib")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "feature": feature,
        "val_acc": results["val"]["acc"],
        "val_macro_f1": results["val"]["macro_f1"],
        "test_acc": results["test"]["acc"],
        "test_macro_f1": results["test"]["macro_f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default=r"data\manifests_source_split")
    ap.add_argument("--out_dir", default=r"reports\feature_ablation")
    ap.add_argument("--features", nargs="+", default=["mfcc", "logmel", "pcenmel", "logstft"])
    args = ap.parse_args()

    mani_dir = ROOT / args.manifest_dir
    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for feat in args.features:
        feat = str(feat).strip().strip("\\/`").lower()
        if not feat:
            continue
        summary.append(run_one(feat, mani_dir, out_root))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")

    print("\n========== SUMMARY ==========")
    print(summary_df)


if __name__ == "__main__":
    main()