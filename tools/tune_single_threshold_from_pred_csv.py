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
        raise RuntimeError(f"Cannot find any of {candidates}. Existing columns={list(df.columns)}")
    return None


def load_pred_csv(path):
    df = pd.read_csv(path)

    prob_col = find_col(
        df,
        ["prob_pos", "p_cough", "prob_cough", "cough_prob", "score_cough", "prob"],
    )

    true_col = find_col(
        df,
        ["true_label", "label", "y_true", "true", "target"],
    )

    out = pd.DataFrame()
    out["true_label"] = df[true_col].astype(str).str.strip().str.lower()
    out["prob_cough"] = pd.to_numeric(df[prob_col], errors="coerce")
    out = out.dropna(subset=["prob_cough"]).copy()
    return out


def encode(labels):
    # 0 = cough, 1 = other
    return np.array([0 if x == "cough" else 1 for x in labels], dtype=np.int64)


def pred_from_prob(prob, thr):
    return np.where(prob >= thr, 0, 1)


def eval_one(y_true, prob, thr):
    y_pred = pred_from_prob(prob, thr)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return acc, mf1, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val = load_pred_csv(args.val_csv)
    test = load_pred_csv(args.test_csv)

    y_val = encode(val["true_label"])
    y_test = encode(test["true_label"])

    thresholds = np.arange(0.30, 0.7001, 0.01)

    rows = []
    best = None

    for thr in thresholds:
        acc, mf1, _ = eval_one(y_val, val["prob_cough"].values, thr)
        row = {
            "thr": float(thr),
            "val_acc": float(acc),
            "val_macro_f1": float(mf1),
        }
        rows.append(row)

        if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
            best = row

    grid = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    grid.to_csv(out_dir / "single_threshold_val_grid.csv", index=False, encoding="utf-8-sig")

    thr = best["thr"]
    test_acc, test_mf1, y_pred = eval_one(y_test, test["prob_cough"].values, thr)

    print("[BEST ON VAL]")
    print(best)

    print("\n[TEST RESULT]")
    print(f"thr={thr:.2f}")
    print(f"ACC={test_acc:.4f}")
    print(f"macro_f1={test_mf1:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    summary = pd.DataFrame([{
        **best,
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_mf1),
    }])
    summary.to_csv(out_dir / "single_threshold_summary.csv", index=False, encoding="utf-8-sig")

    test_out = test.copy()
    test_out["pred_id"] = y_pred
    test_out["pred_label"] = ["cough" if x == 0 else "other" for x in y_pred]
    test_out.to_csv(out_dir / "single_threshold_test_pred.csv", index=False, encoding="utf-8-sig")

    print("\n[OK] wrote ->", out_dir / "single_threshold_summary.csv")


if __name__ == "__main__":
    main()