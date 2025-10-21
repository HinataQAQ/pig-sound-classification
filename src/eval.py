#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained model on a manifest CSV (filepath,label).
Usage:
  python -m src.eval --ckpt checkpoints/model.pt --manifest data/manifests/test.csv --config config/config.yaml
"""
import argparse, yaml, torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
import numpy as np
import pandas as pd

from .dataset import PigAudioDataset
from .models.bilstm import BiLSTMClassifier

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
    labels = cfg["data"]["labels"]
    ds = PigAudioDataset(args.manifest, labels, sr=cfg["data"]["sample_rate"],
                         segment_seconds=cfg["data"]["segment_seconds"],
                         do_denoise=cfg["preprocess"]["do_denoise"],
                         do_vad=cfg["preprocess"]["do_vad"],
                         include_delta=cfg["features"]["include_delta"],
                         include_delta_delta=cfg["features"]["include_delta_delta"],
                         train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = BiLSTMClassifier(input_size=None if "input_size" not in ckpt else ckpt["input_size"],
                             hidden_size=cfg["model"]["hidden_size"],
                             num_layers=cfg["model"]["num_layers"],
                             num_classes=len(labels),
                             dropout=cfg["model"]["dropout"])
    # Infer input size from a batch if needed
    if "input_size" not in ckpt:
        x0, _ = next(iter(dl))
        model = BiLSTMClassifier(input_size=x0.shape[-1],
                                 hidden_size=cfg["model"]["hidden_size"],
                                 num_layers=cfg["model"]["num_layers"],
                                 num_classes=len(labels),
                                 dropout=cfg["model"]["dropout"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    acc = MulticlassAccuracy(num_classes=len(labels))
    f1 = MulticlassF1Score(num_classes=len(labels), average="macro")
    cm = MulticlassConfusionMatrix(num_classes=len(labels))

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            prob = torch.softmax(logits, dim=-1)
            acc.update(prob, y)
            f1.update(prob, y)
            cm.update(prob, y)
            all_probs.append(prob.numpy())
            all_labels.append(y.numpy())
    acc_v = acc.compute().item()
    f1_v = f1.compute().item()
    cm_v = cm.compute().numpy().astype(int)

    print(f"Accuracy={acc_v:.4f}, Macro-F1={f1_v:.4f}")
    print("Confusion matrix (rows=true, cols=pred):\n", cm_v)
    # Save CSV with per-sample predictions (optional)
    probs = np.concatenate(all_probs, axis=0)
    ytrue = np.concatenate(all_labels, axis=0)
    df = pd.DataFrame(probs, columns=[f"prob_{l}" for l in labels])
    df["y_true"] = [labels[i] for i in ytrue]
    df.to_csv("eval_predictions.csv", index=False)
    print("[OK] Wrote eval_predictions.csv")

if __name__ == "__main__":
    main()
