from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


LABELS = ["cough", "other"]


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Cannot find columns {candidates}. Existing columns={list(df.columns)}")
    return None


def load_pred_csv(path):
    df = pd.read_csv(path)

    # 概率列：优先找 prob_pos。你的 eval 脚本终端里一直打印 prob_pos，所以大概率有这列。
    prob_col = find_col(
        df,
        ["prob_pos", "p_cough", "prob_cough", "cough_prob", "score_cough", "prob"],
        required=True,
    )

    true_col = find_col(
        df,
        ["true_label", "label", "y_true", "true", "target"],
        required=True,
    )

    path_col = find_col(
        df,
        ["filepath", "path", "audio_path", "wav_path", "clip_path"],
        required=False,
    )

    out = pd.DataFrame()
    if path_col:
        out["key"] = df[path_col].astype(str)
    else:
        out["key"] = np.arange(len(df)).astype(str)

    out["true_label"] = df[true_col].astype(str).str.strip().str.lower()
    out["prob_cough"] = pd.to_numeric(df[prob_col], errors="coerce")

    out = out.dropna(subset=["prob_cough"]).copy()
    return out


def encode_labels(labels):
    # eval_crnn_logmel 的 --pos_label cough 表示 prob_cough >= threshold 判 cough
    return np.array([0 if str(x).lower() == "cough" else 1 for x in labels], dtype=np.int64)


def pred_from_prob(prob, thr):
    # 0 = cough, 1 = other
    return np.where(prob >= thr, 0, 1)


def eval_prob(y_true, prob, thr):
    y_pred = pred_from_prob(prob, thr)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return acc, mf1, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_a", required=True)
    ap.add_argument("--val_b", required=True)
    ap.add_argument("--test_a", required=True)
    ap.add_argument("--test_b", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_a = load_pred_csv(args.val_a)
    val_b = load_pred_csv(args.val_b)
    test_a = load_pred_csv(args.test_a)
    test_b = load_pred_csv(args.test_b)

    val = val_a.merge(val_b[["key", "prob_cough"]], on="key", suffixes=("_a", "_b"))
    test = test_a.merge(test_b[["key", "prob_cough"]], on="key", suffixes=("_a", "_b"))

    if len(val) == 0:
        raise RuntimeError("Val merge got 0 rows. Check filepath/key columns.")
    if len(test) == 0:
        raise RuntimeError("Test merge got 0 rows. Check filepath/key columns.")

    y_val = encode_labels(val["true_label"])
    y_test = encode_labels(test["true_label"])

    best = None
    rows = []

    weights = np.arange(0.0, 1.0001, 0.05)
    thresholds = np.arange(0.30, 0.7001, 0.01)

    for w in weights:
        prob_val = w * val["prob_cough_a"].values + (1.0 - w) * val["prob_cough_b"].values

        for thr in thresholds:
            acc, mf1, _ = eval_prob(y_val, prob_val, thr)
            rows.append({
                "weight_a_transfer": float(w),
                "weight_b_round4": float(1.0 - w),
                "thr": float(thr),
                "val_acc": float(acc),
                "val_macro_f1": float(mf1),
            })

            score = mf1
            if best is None or score > best["val_macro_f1"]:
                best = rows[-1]

    grid_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    grid_df.to_csv(out_dir / "ensemble_val_grid.csv", index=False, encoding="utf-8-sig")

    print("[BEST ON VAL]")
    print(best)

    w = best["weight_a_transfer"]
    thr = best["thr"]

    prob_test = w * test["prob_cough_a"].values + (1.0 - w) * test["prob_cough_b"].values
    test_acc, test_mf1, y_pred = eval_prob(y_test, prob_test, thr)

    print("\n[TEST RESULT]")
    print(f"ACC={test_acc:.4f}")
    print(f"macro_f1={test_mf1:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    out_pred = test.copy()
    out_pred["ensemble_prob_cough"] = prob_test
    out_pred["ensemble_pred_id"] = y_pred
    out_pred["ensemble_pred_label"] = ["cough" if x == 0 else "other" for x in y_pred]
    out_pred.to_csv(out_dir / "ensemble_test_pred.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        **best,
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_mf1),
    }])
    summary.to_csv(out_dir / "ensemble_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[OK] wrote ->", out_dir / "ensemble_summary.csv")
    print("[OK] wrote ->", out_dir / "ensemble_test_pred.csv")


if __name__ == "__main__":
    main()