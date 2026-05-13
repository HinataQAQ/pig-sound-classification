# src/train_crnn.py
import os
import argparse
import random
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_logmel import LogMelDataset
from src.models.crnn import CRNNClassifier

# Windows OMP 冲突临时绕过（你遇到过 OMP Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/cough_silence_crnn.yaml")
    ap.add_argument("--init_ckpt", type=str, default=None)
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(flag: str):
    if str(flag).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag


@torch.no_grad()
def evaluate(model, dl, device, num_classes: int):
    model.eval()
    all_pred = []
    all_true = []
    total_loss = 0.0
    total_n = 0

    crit = nn.CrossEntropyLoss()

    for x, y in tqdm(dl, desc="[valid]", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = crit(logits, y)

        total_loss += float(loss.detach()) * x.size(0)
        total_n += x.size(0)

        pred = torch.argmax(logits, dim=1)
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    acc = float((y_pred == y_true).mean())

    # macro-f1
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    avg_loss = total_loss / max(1, total_n)
    return acc, macro_f1, float(avg_loss)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(int(cfg["train"]["seed"]))
    device = auto_device(cfg["train"]["device"])
    print("[INFO] device =", device)

    labels = cfg["data"]["labels"]
    num_classes = len(labels)

    logmel_cfg = cfg.get("logmel", {})
    preprocess_cfg = cfg.get("preprocess", {})

    ds_tr = LogMelDataset(
        manifest_path=cfg["data"]["train_manifest"],
        labels=labels,
        sr=int(cfg["data"]["sample_rate"]),
        segment_seconds=float(cfg["data"]["segment_seconds"]),
        n_mels=int(cfg["logmel"]["n_mels"]),
        fmin=float(cfg["logmel"]["fmin"]),
        fmax=float(cfg["logmel"]["fmax"]),
        hop_ms=float(cfg["logmel"]["hop_ms"]),
        win_ms=float(cfg["logmel"]["win_ms"]),
        n_fft=int(cfg["logmel"]["n_fft"]),
        use_specaug=bool(cfg["logmel"].get("use_specaug", False)),
        specaug=cfg["logmel"].get("specaug", {}),
        seed=int(cfg["train"]["seed"]),
        train=True,
        do_denoise=bool(cfg.get("preprocess", {}).get("do_denoise", False)),
        ref_mode=str(cfg["logmel"].get("ref_mode", "fixed")),
        db_ref=float(cfg["logmel"].get("db_ref", 1.0)),
        top_db=float(cfg["logmel"].get("top_db", 80.0)),
        cmvn=str(cfg["logmel"].get("cmvn", "none")),
    )

    ds_va = LogMelDataset(
        manifest_path=cfg["data"]["val_manifest"],
        labels=labels,
        sr=int(cfg["data"]["sample_rate"]),
        segment_seconds=float(cfg["data"]["segment_seconds"]),
        n_mels=int(cfg["logmel"]["n_mels"]),
        fmin=float(cfg["logmel"]["fmin"]),
        fmax=float(cfg["logmel"]["fmax"]),
        hop_ms=float(cfg["logmel"]["hop_ms"]),
        win_ms=float(cfg["logmel"]["win_ms"]),
        n_fft=int(cfg["logmel"]["n_fft"]),
        use_specaug=False,
        seed=int(cfg["train"]["seed"]),
        train=False,
        do_denoise=bool(cfg.get("preprocess", {}).get("do_denoise", False)),
        ref_mode=str(cfg["logmel"].get("ref_mode", "fixed")),
        db_ref=float(cfg["logmel"].get("db_ref", 1.0)),
        top_db=float(cfg["logmel"].get("top_db", 80.0)),
        cmvn=str(cfg["logmel"].get("cmvn", "none")),
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )

    model_cfg = cfg.get("model", {})
    model = CRNNClassifier(
        n_mels=int(logmel_cfg.get("n_mels", 64)),
        num_classes=num_classes,
        cnn_channels=tuple(model_cfg.get("cnn_channels", (16, 32, 64))),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 128)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        src_state = ckpt["state_dict"]
        tgt_state = model.state_dict()

        usable = {
            k: v for k, v in src_state.items()
            if k in tgt_state and tgt_state[k].shape == v.shape
        }

        model.load_state_dict(usable, strict=False)
        print(f"[INIT] loaded pretrained weights from {args.init_ckpt}")
        print(f"[INIT] usable tensors = {len(usable)}")

    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    exp = cfg.get("experiment_name", "exp")
    os.makedirs("checkpoints", exist_ok=True)
    best_acc = -1.0
    best_path = os.path.join("checkpoints", f"{exp}_best.pt")

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}/{epochs} [train]")
        loss_avg = 0.0

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()

            gc = float(cfg["train"].get("grad_clip", 0.0))
            if gc and gc > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gc)

            opt.step()
            loss_avg = loss_avg * 0.9 + float(loss.detach()) * 0.1
            pbar.set_postfix(loss=f"{loss_avg:.4f}")

        acc, f1, vloss = evaluate(model, dl_va, device, num_classes)
        print(f"Epoch {epoch} | val acc={acc:.4f} f1={f1:.4f} val_loss={vloss:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "config": cfg,
                },
                best_path,
            )
            print(f"[+] Saved best -> {best_path} (acc={best_acc:.4f})")

    print("[DONE] best_acc =", best_acc)


if __name__ == "__main__":
    main()
