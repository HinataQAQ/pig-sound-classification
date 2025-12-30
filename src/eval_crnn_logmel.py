# src/eval_crnn_logmel.py
import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

from .utils.audio_io import load_wav, fix_length
from .preprocessing.denoise import spectral_subtraction
from .models.crnn import CRNNClassifier


def compute_logmel(
    y, sr,
    n_mels=64, fmin=50, fmax=8000,
    hop_ms=10.0, win_ms=25.0, n_fft=1024,
    db_ref="max", top_db=80.0
):
    hop_length = int(sr * hop_ms / 1000.0)
    win_length = int(sr * win_ms / 1000.0)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
        power=2.0, center=False
    )

    # 关键：ref 不要默认 np.max（那会把静音也“抬到 0dB”）
    if isinstance(db_ref, str) and db_ref.lower() == "max":
        ref = np.max
    else:
        ref = float(db_ref)

    logmel = librosa.power_to_db(mel, ref=ref, top_db=float(top_db)).T  # (T,F)
    return logmel.astype(np.float32)



class LogMelClipDataset(Dataset):
    def __init__(self, manifest_csv, labels, sr=32000, segment_seconds=1.0,
                 do_denoise=False, logmel_cfg=None):
        self.df = pd.read_csv(manifest_csv)
        self.labels = [str(l).strip().lower() for l in labels]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.sr = int(sr)
        self.seg_len = int(self.sr * float(segment_seconds))
        self.do_denoise = bool(do_denoise)
        self.logmel_cfg = logmel_cfg or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        lab = str(row["label"]).strip().lower()
        if lab not in self.label2id:
            raise KeyError(f"label '{lab}' not in {self.labels}")
        y_true = self.label2id[lab]

        y, sr = load_wav(path, sr=self.sr, mono=True)
        if self.do_denoise:
            y = spectral_subtraction(y, sr)

        # 这里 manifest 的切片一般就是 1s，所以直接截/补到固定长度即可
        if len(y) < self.seg_len:
            y = fix_length(y, self.seg_len)
        else:
            y = y[:self.seg_len]

        x = compute_logmel(
            y, sr=self.sr,
            n_mels=int(self.logmel_cfg.get("n_mels", 64)),
            fmin=float(self.logmel_cfg.get("fmin", 50)),
            fmax=float(self.logmel_cfg.get("fmax", 8000)),
            hop_ms=float(self.logmel_cfg.get("hop_ms", 10.0)),
            win_ms=float(self.logmel_cfg.get("win_ms", 25.0)),
            n_fft=int(self.logmel_cfg.get("n_fft", 1024)),
            db_ref=self.logmel_cfg.get("db_ref", "max"),
            top_db=self.logmel_cfg.get("top_db", 80.0),
        )

        return torch.from_numpy(x), torch.tensor(y_true).long(), path


def pad_collate(batch):
    xs, ys, paths = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    T = max(lengths)
    Fdim = xs[0].shape[1]
    out = torch.zeros(len(xs), T, Fdim, dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, :x.shape[0], :] = x
    return out, torch.stack(list(ys)), list(paths)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_csv", type=str, default="eval_crnn_predictions.csv")
    # 新增：阈值（二分类时非常实用）
    ap.add_argument("--thr", type=float, default=None,
                    help="If set, use prob(pos_label) >= thr => pos_label else other_label.")
    ap.add_argument("--pos_label", type=str, default="cough",
                    help="Positive label name for thresholding.")
    return ap.parse_args()


def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    labels = ckpt.get("labels", cfg["data"]["labels"])
    labels = [str(x).strip().lower() for x in labels]
    id2label = {i: l for i, l in enumerate(labels)}

    # 二分类阈值逻辑需要确定 pos/neg 的 id
    pos = str(args.pos_label).strip().lower()
    if pos not in labels:
        raise ValueError(f"--pos_label={pos} not in labels={labels}")
    pos_id = labels.index(pos)
    if len(labels) != 2:
        raise ValueError(f"threshold mode expects 2 classes, got labels={labels}")
    neg_id = 1 - pos_id

    ds = LogMelClipDataset(
        args.manifest,
        labels=labels,
        sr=cfg["data"].get("sample_rate", 32000),
        segment_seconds=cfg["data"].get("segment_seconds", 1.0),
        do_denoise=cfg.get("preprocess", {}).get("do_denoise", False),
        logmel_cfg=cfg.get("logmel", {}),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=pad_collate)

    model = CRNNClassifier(
        n_mels=int(cfg.get("logmel", {}).get("n_mels", 64)),
        num_classes=len(labels)
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("[DEBUG] missing keys =", missing)
    print("[DEBUG] unexpected keys =", unexpected)
    model.eval()

    rows = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)                  # (B,2)
            probs = torch.softmax(logits, dim=1)  # (B,2)

            if args.thr is None:
                pred_id = probs.argmax(dim=1)     # (B,)
            else:
                prob_pos = probs[:, pos_id]
                pred_id = torch.where(
                    prob_pos >= float(args.thr),
                    torch.full_like(prob_pos, pos_id, dtype=torch.long),
                    torch.full_like(prob_pos, neg_id, dtype=torch.long),
                )

            pred_np = pred_id.cpu().numpy()
            y_np = y.cpu().numpy()

            correct += int((pred_np == y_np).sum())
            total += len(paths)

            for i in range(len(paths)):
                rec = {
                    "filepath": paths[i],
                    "y_true": id2label[int(y_np[i])],
                    "y_pred": id2label[int(pred_np[i])],
                }
                for k, lab in enumerate(labels):
                    rec[f"prob_{lab}"] = float(probs[i, k].item())
                rows.append(rec)

    acc = correct / max(1, total)
    print(f"[OK] ACC={acc:.4f} ({correct}/{total})")

    out_path = os.path.abspath(args.out_csv)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("[CONFUSION]\n", pd.crosstab(df_out["y_true"], df_out["y_pred"], rownames=["true"], colnames=["pred"]))
    print("[OK] wrote ->", out_path)


if __name__ == "__main__":
    main()
