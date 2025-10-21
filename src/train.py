import os, argparse, yaml, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm

from .dataset import PigAudioDataset
from .models.bilstm import BiLSTMClassifier

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    return ap.parse_args()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def auto_device(flag):
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag

def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["train"]["seed"])
    device = auto_device(cfg["train"]["device"])

    labels = cfg["data"]["labels"]
    ds_tr = PigAudioDataset(cfg["data"]["train_manifest"], labels, sr=cfg["data"]["sample_rate"],
                            segment_seconds=cfg["data"]["segment_seconds"],
                            do_denoise=cfg["preprocess"]["do_denoise"],
                            do_vad=cfg["preprocess"]["do_vad"],
                            include_delta=cfg["features"]["include_delta"],
                            include_delta_delta=cfg["features"]["include_delta_delta"],
                            train=True)
    ds_va = PigAudioDataset(cfg["data"]["val_manifest"], labels, sr=cfg["data"]["sample_rate"],
                            segment_seconds=cfg["data"]["segment_seconds"],
                            do_denoise=cfg["preprocess"]["do_denoise"],
                            do_vad=cfg["preprocess"]["do_vad"],
                            include_delta=cfg["features"]["include_delta"],
                            include_delta_delta=cfg["features"]["include_delta_delta"],
                            train=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    xb, yb = next(iter(dl_tr))
    input_size = xb.shape[-1]

    model = BiLSTMClassifier(input_size=input_size,
                             hidden_size=cfg["model"]["hidden_size"],
                             num_layers=cfg["model"]["num_layers"],
                             num_classes=len(labels),
                             dropout=cfg["model"]["dropout"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    acc_metric = MulticlassAccuracy(num_classes=len(labels)).to(device)
    f1_metric = MulticlassF1Score(num_classes=len(labels), average="macro").to(device)

    best_va = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}/{cfg['train']['epochs']} [train]")
        loss_avg = 0.0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg["train"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            loss_avg = loss_avg * 0.9 + float(loss) * 0.1
            pbar.set_postfix(loss=f"{loss_avg:.4f}")

        # validate
        model.eval()
        acc_metric.reset(); f1_metric.reset()
        with torch.no_grad():
            for x, y in tqdm(dl_va, desc="[valid]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                acc_metric.update(logits, y)
                f1_metric.update(logits, y)
        acc = acc_metric.compute().item()
        f1 = f1_metric.compute().item()
        print(f"Epoch {epoch} | val acc={acc:.4f} f1={f1:.4f}")
        if acc > best_va:
            best_va = acc
            torch.save({"state_dict": model.state_dict(), "labels": labels, "input_size": input_size}, "checkpoints/model.pt")
            print(f"[+] Saved best model to checkpoints/model.pt (acc={best_va:.4f})")

if __name__ == "__main__":
    main()
