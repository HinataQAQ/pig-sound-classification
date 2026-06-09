from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["cough", "calm_grunt", "feeding", "stress_vocal"]


def infer_path_col(df):
    for c in ["path", "filepath", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column from columns={list(df.columns)}")


def infer_label_col(df):
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column from columns={list(df.columns)}")


def load_duration_features(csv_path):
    df = pd.read_csv(csv_path)
    path_col = infer_path_col(df)
    label_col = infer_label_col(df)

    xs = []
    ys = []

    for _, r in df.iterrows():
        p = ROOT / str(r[path_col]).replace("\\", "/")
        lab = str(r[label_col]).strip().lower()

        info = sf.info(str(p))
        dur = float(info.frames) / float(info.samplerate)

        xs.append([dur])
        ys.append(LABELS.index(lab))

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def run_fold(fold):
    base = ROOT / "data" / "manifests_pigvocal_4class_expanded_train_cv5_cap3x" / f"fold{fold}"

    x_train, y_train = load_duration_features(base / "train.csv")
    x_test, y_test = load_duration_features(base / "test.csv")

    models = {
        "duration_logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "duration_tree": DecisionTreeClassifier(max_depth=3, random_state=3407, class_weight="balanced"),
    }

    rows = []

    for name, clf in models.items():
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro", labels=list(range(len(LABELS))), zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=list(range(len(LABELS))))

        print(f"\n[FOLD {fold}] {name}")
        print(f"ACC={acc:.4f} macro_f1={mf1:.4f}")
        print(classification_report(y_test, pred, target_names=LABELS, digits=4, zero_division=0))
        print(cm)

        rows.append({
            "fold": fold,
            "model": name,
            "acc": acc,
            "macro_f1": mf1,
            "confusion": cm.tolist(),
        })

    return rows


def main():
    all_rows = []

    for fold in range(5):
        all_rows.extend(run_fold(fold))

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "confusion"}
        for r in all_rows
    ])

    summary = (
        df.groupby("model")
        .agg(
            n=("macro_f1", "count"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
        )
        .reset_index()
        .sort_values("mean_macro_f1", ascending=False)
    )

    out_runs = ROOT / "reports" / "duration_only_baseline_runs.csv"
    out_summary = ROOT / "reports" / "duration_only_baseline_summary.csv"

    df.to_csv(out_runs, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("\n[DURATION-ONLY SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[OK] wrote -> {out_runs}")
    print(f"[OK] wrote -> {out_summary}")


if __name__ == "__main__":
    main()