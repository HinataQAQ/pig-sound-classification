#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained model on a manifest CSV (filepath,label).
Usage:
  python -m src.eval --ckpt checkpoints/model.pt --manifest data/manifests/test.csv --config config/config.yaml
"""
import argparse, yaml, torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.dataset import PigAudioDataset
from src.models.bilstm import BiLSTMClassifier

try:
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
    _HAVE_TORCHMETRICS = True
except Exception:
    _HAVE_TORCHMETRICS = False

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to checkpoints/model.pt")
    ap.add_argument("--manifest", required=True, help="CSV with filepath,label")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--batch_size", type=int, default=32)
    return ap.parse_args()

def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = cfg["data"]["labels"]
    num_classes = len(labels)
    ds = PigAudioDataset(args.manifest, labels,
                         sr=cfg["data"]["sample_rate"],
                         segment_seconds=cfg["data"]["segment_seconds"],
                         do_denoise=cfg["preprocess"]["do_denoise"],
                         do_vad=cfg["preprocess"]["do_vad"],
                         include_delta=cfg["features"]["include_delta"],
                         include_delta_delta=cfg["features"]["include_delta_delta"],
                         train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    input_size = ckpt.get("input_size", None)
    if input_size is None:
        x0, _ = next(iter(dl))
        input_size = x0.shape[-1]

    model = BiLSTMClassifier(input_size=input_size,
                             hidden_size=cfg["model"]["hidden_size"],
                             num_layers=cfg["model"]["num_layers"],
                             num_classes=num_classes,
                             dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # --- metrics ---
    use_multiclass = (num_classes > 1) and _HAVE_TORCHMETRICS
    if use_multiclass:
        acc = MulticlassAccuracy(num_classes=num_classes).to(device)
        f1  = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        cm  = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    else:
        acc = f1 = cm = None
        print("[warn] single-class dataset or torchmetrics missing: disable multiclass metrics; will report manual accuracy only.")

    all_probs = []
    all_labels = []
    manual_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            # 统一用 softmax 概率；单类时会得到全 1 的列，我们单独处理
            if num_classes > 1:
                prob = torch.softmax(logits, dim=-1)
            else:
                # 单类模型：输出 shape=(B,1)，视为该类概率恒为 1
                prob = torch.ones((logits.shape[0], 1), device=logits.device)

            if use_multiclass:
                acc.update(prob, y.to(device))
                f1.update(prob, y.to(device))
                cm.update(prob, y.to(device))
            else:
                pred_idx = prob.argmax(dim=-1).cpu()
                manual_correct += (pred_idx == y).sum().item()
                total += y.shape[0]

            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.numpy())

    # 汇总并导出
    probs = np.concatenate(all_probs, axis=0)
    ytrue = np.concatenate(all_labels, axis=0)
    df = pd.DataFrame(probs, columns=[f"prob_{l}" for l in labels])
    df["y_true"] = [labels[i] for i in ytrue]
    df.to_csv("eval_predictions.csv", index=False)
    print("[OK] Wrote eval_predictions.csv")

    if use_multiclass:
        acc_v = acc.compute().item()
        f1_v  = f1.compute().item()
        print(f"ACC={acc_v:.4f}, Macro-F1={f1_v:.4f}")
        # 如需混淆矩阵：
        # cm_v = cm.compute().cpu().numpy().astype(int)
        # print(cm_v)
    else:
        manual_acc = manual_correct / max(1, total)
        print(f"[single-class] accuracy={manual_acc:.4f}  (metrics disabled)")

if __name__ == "__main__":
    main()