from pathlib import Path
import json
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
MANI_DIR = ROOT / "data" / "manifests_korea_subtype_clean"
OUT_DIR = ROOT / "reports" / "korea_mfcc_svm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID = {
    "dry_cough": 0,
    "abdominal_cough": 1,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SR = 32000
N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = int(SR * 0.010)
WIN_LENGTH = int(SR * 0.025)

def extract_feature(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta1.mean(axis=1), delta1.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ], axis=0)

    return feat.astype(np.float32)

def load_split(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)
    X, y = [], []
    bad_rows = []

    for _, row in df.iterrows():
        p = str(row["filepath"]).strip()
        lab = str(row["label"]).strip()

        try:
            x = extract_feature(p)
            X.append(x)
            y.append(LABEL2ID[lab])
        except Exception as e:
            bad_rows.append({
                "filepath": p,
                "label": lab,
                "error": repr(e),
            })
            print(f"[BAD][{split_name}] {p} -> {repr(e)}")

    bad_df = pd.DataFrame(bad_rows)
    bad_df.to_csv(OUT_DIR / f"{split_name}_bad_audio.csv", index=False, encoding="utf-8-sig")

    print(f"[INFO] {split_name}: usable={len(X)} bad={len(bad_rows)}")

    return np.stack(X), np.array(y, dtype=np.int64)

print("[INFO] loading train...")
X_train, y_train = load_split(MANI_DIR / "train.csv", "train")
print("[INFO] loading val...")
X_val, y_val = load_split(MANI_DIR / "val.csv", "val")
print("[INFO] loading test...")
X_test, y_test = load_split(MANI_DIR / "test.csv", "test")

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=3407)),
])

print("[INFO] fitting SVM...")
clf.fit(X_train, y_train)

for split_name, X, y_true in [
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n[{split_name.upper()}] ACC={acc:.4f} macro_f1={macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["dry_cough", "abdominal_cough"], digits=4))
    print(confusion_matrix(y_true, y_pred))

    out_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": [ID2LABEL[int(v)] for v in y_true],
        "pred_label": [ID2LABEL[int(v)] for v in y_pred],
    })
    out_df.to_csv(OUT_DIR / f"{split_name}_pred.csv", index=False, encoding="utf-8-sig")

joblib.dump(clf, OUT_DIR / "mfcc_svm.joblib")

with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "sr": SR,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
    }, f, ensure_ascii=False, indent=2)

print("[OK] saved model ->", OUT_DIR / "mfcc_svm.joblib")