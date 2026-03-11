# src/eval_crnn_logmel.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset_logmel import LogMelDataset
from .models.crnn import CRNNClassifier


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_csv", type=str, default="eval_crnn_predictions.csv")
    ap.add_argument("--thr", type=float, default=None, help="prob(pos_label)>=thr => pos_label (binary only)")
    ap.add_argument("--pos_label", type=str, default="cough")
    return ap.parse_args()


def main():
    args = parse()
    cfg_file = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)

    ckpt = torch.load(args.ckpt, map_location=device)

    # 用 ckpt 里保存的 config 优先（避免你改了 yaml 导致结构不一致）
    cfg = ckpt.get("config", cfg_file)
    if not isinstance(cfg, dict):
        cfg = cfg_file

    labels = ckpt.get("labels", cfg["data"]["labels"])
    labels_lower = [str(x).lower() for x in labels]

    ds = LogMelDataset(
        manifest_path=args.manifest,
        labels=labels,
        sr=int(cfg["data"].get("sample_rate", 32000)),
        segment_seconds=float(cfg["data"].get("segment_seconds", 1.0)),
        n_mels=int(cfg.get("logmel", {}).get("n_mels", 64)),
        fmin=float(cfg.get("logmel", {}).get("fmin", 50)),
        fmax=float(cfg.get("logmel", {}).get("fmax", 8000)),
        hop_ms=float(cfg.get("logmel", {}).get("hop_ms", 10.0)),
        win_ms=float(cfg.get("logmel", {}).get("win_ms", 25.0)),
        n_fft=int(cfg.get("logmel", {}).get("n_fft", 1024)),
        use_specaug=False,
        seed=int(cfg.get("train", {}).get("seed", 3407)),
        train=False,
        do_denoise=bool(cfg.get("preprocess", {}).get("do_denoise", False)),
        ref_mode=str(cfg.get("logmel", {}).get("ref_mode", "fixed")),
        db_ref=float(cfg.get("logmel", {}).get("db_ref", 1.0)),
        top_db=float(cfg.get("logmel", {}).get("top_db", 80.0)),
        cmvn=str(cfg.get("logmel", {}).get("cmvn", "none")),
        return_path=True,
    )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mcfg = cfg.get("model", {})
    model = CRNNClassifier(
        n_mels=int(cfg.get("logmel", {}).get("n_mels", 64)),
        num_classes=len(labels),
        cnn_channels=tuple(mcfg.get("cnn_channels", [16, 32, 64])),
        rnn_hidden=int(mcfg.get("rnn_hidden", 128)),
        rnn_layers=int(mcfg.get("rnn_layers", 2)),
        dropout=float(mcfg.get("dropout", 0.0)),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("[DEBUG] missing keys =", missing)
    print("[DEBUG] unexpected keys =", unexpected)
    model.eval()

    # pos_label 索引
    pos_label = str(args.pos_label).lower()
    pos_idx = labels_lower.index(pos_label) if pos_label in labels_lower else 0
    if len(labels) == 2:
        neg_idx = 1 - pos_idx
    else:
        neg_idx = None

    id2label = {i: l for i, l in enumerate(labels)}
    rows = []
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (B,C)
            probs = torch.softmax(logits, dim=1)  # (B,C)

            if args.thr is not None and len(labels) == 2:
                p_pos = probs[:, pos_idx]
                pred_id = torch.where(
                    p_pos >= float(args.thr),
                    torch.tensor(pos_idx, device=device),
                    torch.tensor(neg_idx, device=device),
                )
            else:
                pred_id = probs.argmax(dim=1)

            y_cpu = y.cpu().numpy()
            pred_cpu = pred_id.cpu().numpy()

            for i in range(len(paths)):
                rec = {
                    "filepath": paths[i],
                    "y_true": id2label[int(y_cpu[i])],
                    "y_pred": id2label[int(pred_cpu[i])],
                }
                for k, lab in enumerate(labels):
                    rec[f"prob_{lab}"] = float(probs[i, k].cpu().item())
                rows.append(rec)

            y_true_all.extend(list(y_cpu))
            y_pred_all.extend(list(pred_cpu))

    y_true_all = np.array(y_true_all, dtype=np.int64)
    y_pred_all = np.array(y_pred_all, dtype=np.int64)
    acc = float((y_true_all == y_pred_all).mean())
    print(f"[OK] ACC={acc:.4f} ({int((y_true_all == y_pred_all).sum())}/{len(y_true_all)})")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("[OK] wrote ->", os.path.abspath(args.out_csv))

    # 混淆矩阵
    print("\n[CONFUSION]\n", pd.crosstab(df["y_true"], df["y_pred"], rownames=["true"], colnames=["pred"]))

    if len(labels) == 2 and f"prob_{labels[pos_idx]}" in df.columns:
        print("\n== prob_pos mean by true ==")
        print(df.groupby("y_true")[f"prob_{labels[pos_idx]}"].mean().to_dict())


if __name__ == "__main__":
    main()
