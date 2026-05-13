import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import PigAudioDataset
from src.models.bilstm import BiLSTMClassifier


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="checkpoints/model.pt")
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--batch_size", type=int, default=64)
    return ap.parse_args()


def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ckpt = torch.load(args.model, map_location="cpu")

    labels = [str(x).strip().lower() for x in ckpt.get("labels", cfg["data"]["labels"])]

    manifest = cfg["data"]["test_manifest"] if args.split == "test" else cfg["data"]["val_manifest"]

    ds = PigAudioDataset(
        manifest,
        labels,
        sr=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        do_denoise=cfg["preprocess"]["do_denoise"],
        do_vad=cfg["preprocess"]["do_vad"],
        include_delta=cfg["features"]["include_delta"],
        include_delta_delta=cfg["features"]["include_delta_delta"],
        train=False,
        mfcc_config=cfg["features"].get("mfcc", {}),
        gfcc_config=cfg["features"].get("gfcc", {}),
        vad_config=cfg.get("preprocess", {}).get("vad", {}),
        cmvn=cfg["features"].get("cmvn", "utt"),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        input_size=int(ckpt["input_size"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_classes=len(labels),
        dropout=float(cfg["model"]["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())

    print(f"\n=== {args.split.upper()} REPORT ===")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print("balanced_acc =", balanced_accuracy_score(y_true, y_pred))
    print("confusion_matrix =")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()