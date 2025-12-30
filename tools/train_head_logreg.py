# tools/train_head_logreg.py
import argparse
from pathlib import Path
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    labels = list(d["labels"]) if "labels" in d.files else None
    return X, y, labels

def eval_with_threshold(y_true, cough_prob, thr, cough_id=0, other_id=1):
    y_pred = np.where(cough_prob >= thr, cough_id, other_id).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m, y_pred

def search_best_threshold(y_true, cough_prob, metric="f1", cough_id=0, other_id=1):
    best = (-1.0, -1.0, 0.5)  # (acc, f1, thr)
    for thr in np.linspace(0.05, 0.95, 91):
        acc, f1m, _ = eval_with_threshold(y_true, cough_prob, thr, cough_id, other_id)
        score = f1m if metric == "f1" else acc
        best_score = best[1] if metric == "f1" else best[0]
        if score > best_score:
            best = (acc, f1m, float(thr))
    return best  # (acc, f1, thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--out", default="checkpoints/head_logreg.npz")
    ap.add_argument("--cough_id", type=int, default=0)
    ap.add_argument("--other_id", type=int, default=1)
    args = ap.parse_args()

    Xtr, ytr, labels = load_npz(args.train)
    Xva, yva, labels2 = load_npz(args.val)
    if labels is None and labels2 is not None:
        labels = labels2
    if labels is None:
        labels = ["cough", "other"]

    # 线性头：标准化 + 逻辑回归
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="liblinear",   # 小数据集常用
            class_weight="balanced"
        ))
    ])

    clf.fit(Xtr, ytr)

    # 获取 cough 概率（注意 classes_ 的顺序）
    proba_va = clf.predict_proba(Xva)
    classes = clf.named_steps["lr"].classes_
    idx_cough = int(np.where(classes == args.cough_id)[0][0])
    cough_prob_va = proba_va[:, idx_cough]

    # 在 val 上找最佳阈值（默认优化 macro-F1）
    best_acc, best_f1, best_thr = search_best_threshold(
        yva, cough_prob_va, metric="f1", cough_id=args.cough_id, other_id=args.other_id
    )
    acc05, f105, ypred05 = eval_with_threshold(yva, cough_prob_va, 0.5, args.cough_id, args.other_id)
    _, _, ypred_best = eval_with_threshold(yva, cough_prob_va, best_thr, args.cough_id, args.other_id)

    print("=== VAL (thr=0.50) ===")
    print(f"ACC={acc05:.4f}  Macro-F1={f105:.4f}")
    print(confusion_matrix(yva, ypred05, labels=[args.cough_id, args.other_id]))
    print(classification_report(yva, ypred05, target_names=labels, digits=4))

    print("\n=== VAL (best_thr by Macro-F1) ===")
    print(f"best_thr={best_thr:.2f}  ACC={best_acc:.4f}  Macro-F1={best_f1:.4f}")
    print(confusion_matrix(yva, ypred_best, labels=[args.cough_id, args.other_id]))
    print(classification_report(yva, ypred_best, target_names=labels, digits=4))

    # 保存（用 npz 保存最简单，不依赖 joblib）
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, coef=clf.named_steps["lr"].coef_, intercept=clf.named_steps["lr"].intercept_,
                        classes=classes, mean=clf.named_steps["scaler"].mean_, scale=clf.named_steps["scaler"].scale_,
                        labels=np.array(labels, dtype=object), best_thr=best_thr)
    print(f"\n[OK] saved head -> {out}")

    # 可选：在 test_200 上也跑一下（用 best_thr）
    if args.test:
        Xte, yte, _ = load_npz(args.test)
        proba_te = clf.predict_proba(Xte)
        cough_prob_te = proba_te[:, idx_cough]
        acc_te, f1_te, ypred_te = eval_with_threshold(yte, cough_prob_te, best_thr, args.cough_id, args.other_id)
        print("\n=== TEST (using best_thr from VAL) ===")
        print(f"ACC={acc_te:.4f}  Macro-F1={f1_te:.4f}")
        print(confusion_matrix(yte, ypred_te, labels=[args.cough_id, args.other_id]))

if __name__ == "__main__":
    main()
