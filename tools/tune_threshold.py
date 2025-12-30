# tools/tune_threshold.py
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--positive", default="cough")
    ap.add_argument("--steps", type=int, default=400)   # 阈值扫描密度
    ap.add_argument("--pick", choices=["macro_f1", "accuracy", "precision90"], default="macro_f1")
    ap.add_argument("--out_thr", default="checkpoints/best_threshold.txt")
    args = ap.parse_args()

    dfv = pd.read_csv(args.val_csv)
    assert "prob_cough" in dfv.columns and "y_true" in dfv.columns, f"bad columns: {list(dfv.columns)}"

    y_true = (dfv["y_true"].astype(str) == args.positive).astype(int).to_numpy()
    score = dfv["prob_cough"].to_numpy()

    thresholds = np.linspace(0.0, 1.0, args.steps + 1)

    best = None
    for thr in thresholds:
        pred = (score >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)

        # precision / recall（手算避免依赖更多包）
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if args.pick == "macro_f1":
            key = f1
        elif args.pick == "accuracy":
            key = acc
        else:  # precision90：优先满足 precision>=0.90，其次最大化 recall
            key = (1.0 if prec >= 0.90 else 0.0, rec)

        if best is None or key > best["key"]:
            best = dict(key=key, thr=float(thr), acc=float(acc), f1=float(f1), prec=float(prec), rec=float(rec))

    print("\n=== VAL BEST ===")
    print(f"pick={args.pick}  thr={best['thr']:.4f}  ACC={best['acc']:.4f}  F1={best['f1']:.4f}  P={best['prec']:.4f}  R={best['rec']:.4f}")

    with open(args.out_thr, "w", encoding="utf-8") as f:
        f.write(str(best["thr"]))
    print(f"[OK] saved thr -> {args.out_thr}")

    # 评估 test（如果给了）
    if args.test_csv:
        dft = pd.read_csv(args.test_csv)
        yt = (dft["y_true"].astype(str) == args.positive).astype(int).to_numpy()
        st = dft["prob_cough"].to_numpy()
        pt = (st >= best["thr"]).astype(int)

        print("\n=== TEST (using VAL best thr) ===")
        print("ACC=", accuracy_score(yt, pt))
        print("F1 =", f1_score(yt, pt, zero_division=0))
        print("confusion:\n", confusion_matrix(yt, pt))
        print("\nreport:\n", classification_report(yt, pt, target_names=["other", args.positive], zero_division=0))

if __name__ == "__main__":
    main()
