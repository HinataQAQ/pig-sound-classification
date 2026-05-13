from pathlib import Path
from math import gcd
import argparse
import random
import json

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_path_col(df):
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column. columns={list(df.columns)}")


def infer_label_col(df):
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column. columns={list(df.columns)}")


def load_audio_soundfile(path, sr=32000, dur_s=1.0):
    path = str(Path(path))

    y, s = sf.read(path, dtype="float32", always_2d=False)

    if y is None or len(y) == 0:
        raise RuntimeError(f"zero-length audio: {path}")

    if isinstance(y, np.ndarray) and y.ndim == 2:
        y = y.mean(axis=1)

    y = np.asarray(y, dtype=np.float32)

    if int(s) != int(sr):
        g = gcd(int(s), int(sr))
        up = int(sr) // g
        down = int(s) // g
        y = resample_poly(y, up, down).astype(np.float32)
        s = int(sr)

    target_len = int(sr * dur_s)

    if len(y) < target_len:
        pad = target_len - len(y)
        left = pad // 2
        right = pad - left
        y = np.pad(y, (left, right), mode="constant")
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y.astype(np.float32), int(s)


def norm_map(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / (sd + eps)).astype(np.float32)


def align_rows(mat, target_rows=64):
    """
    把 MFCC / Delta 等 [R, T] 特征插值到 [64, T]，
    这样才能和 Log-Mel 的 64 mel bins 对齐。
    """
    mat = np.asarray(mat, dtype=np.float32)
    r, t = mat.shape

    if r == target_rows:
        return mat.astype(np.float32)

    old_x = np.linspace(0.0, 1.0, r)
    new_x = np.linspace(0.0, 1.0, target_rows)

    out = np.zeros((target_rows, t), dtype=np.float32)

    for j in range(t):
        out[:, j] = np.interp(new_x, old_x, mat[:, j])

    return out.astype(np.float32)


def make_mctafd_feature(
    y,
    sr=32000,
    n_mels=64,
    n_mfcc=20,
    n_fft=1024,
    hop_length=320,
    win_length=800,
    fmin=50,
    fmax=8000,
):
    """
    输出形状：[5, 64, T]

    Channel 0: Log-Mel
    Channel 1: MFCC aligned
    Channel 2: Delta-MFCC + Delta2 aligned
    Channel 3: Transient / positive spectral flux map
    Channel 4: Mel-Cepstral difference map
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)

    # MFCC from Log-Mel representation
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc).astype(np.float32)

    # Delta features
    # mode='nearest' 避免很短样本时插值边界问题
    d1 = librosa.feature.delta(mfcc, width=9, order=1, axis=-1, mode="nearest").astype(np.float32)
    d2 = librosa.feature.delta(mfcc, width=9, order=2, axis=-1, mode="nearest").astype(np.float32)

    mfcc_align = align_rows(mfcc, target_rows=n_mels)
    dyn_align = align_rows(np.concatenate([d1, d2], axis=0), target_rows=n_mels)

    # Transient / spectral flux map
    diff_t = np.diff(logmel, axis=1)
    flux = np.maximum(diff_t, 0.0)
    flux = np.concatenate([np.zeros((n_mels, 1), dtype=np.float32), flux], axis=1)

    # Difference map
    logmel_n = norm_map(logmel)
    mfcc_n = norm_map(mfcc_align)
    diff_map = np.abs(logmel_n - mfcc_n).astype(np.float32)

    x = np.stack(
        [
            norm_map(logmel),
            norm_map(mfcc_align),
            norm_map(dyn_align),
            norm_map(flux),
            norm_map(diff_map),
        ],
        axis=0,
    ).astype(np.float32)

    return x


class MCTAFDDataset(Dataset):
    def __init__(self, manifest, labels, sr=32000, dur_s=1.0, cache=True):
        self.df = pd.read_csv(manifest)
        self.path_col = infer_path_col(self.df)
        self.label_col = infer_label_col(self.df)
        self.labels = labels
        self.label2id = {x: i for i, x in enumerate(labels)}
        self.sr = sr
        self.dur_s = dur_s
        self.cache = cache
        self._cache = {}

        self.df[self.label_col] = self.df[self.label_col].astype(str).str.strip().str.lower()

        keep = self.df[self.label_col].isin(self.label2id.keys())
        bad = self.df[~keep]

        if len(bad) > 0:
            print("[WARN] dropping rows with unknown labels:")
            print(bad[self.label_col].value_counts())

        self.df = self.df[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        label = str(row[self.label_col]).strip().lower()

        y, sr = load_audio_soundfile(path, sr=self.sr, dur_s=self.dur_s)
        x = make_mctafd_feature(y, sr=sr)

        x = torch.from_numpy(x).float()
        yid = torch.tensor(self.label2id[label], dtype=torch.long)

        item = (x, yid)

        if self.cache:
            self._cache[idx] = item

        return item


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class MCTAFDCRNN(nn.Module):
    def __init__(self, num_classes, in_channels=5, fusion_channels=16, cnn_channels=(32, 64, 128), rnn_hidden=128):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            SEBlock(fusion_channels, reduction=4),
        )

        blocks = []
        c_in = fusion_channels

        for c_out in cnn_channels:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ]
            c_in = c_out

        self.cnn = nn.Sequential(*blocks)

        self.rnn = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden * 2, num_classes),
        )

    def forward(self, x):
        # x: [B, 5, 64, T]
        x = self.frontend(x)
        x = self.cnn(x)  # [B, C, F, T]

        # frequency average -> temporal sequence
        x = x.mean(dim=2)      # [B, C, T]
        x = x.permute(0, 2, 1) # [B, T, C]

        out, _ = self.rnn(x)   # [B, T, 2H]

        # temporal average pooling
        z = out.mean(dim=1)

        logits = self.classifier(z)
        return logits


def compute_class_weights(ds, labels):
    counts = np.zeros(len(labels), dtype=np.float32)
    for _, row in ds.df.iterrows():
        lab = str(row[ds.label_col]).strip().lower()
        counts[ds.label2id[lab]] += 1

    weights = counts.sum() / (len(labels) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32), counts


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()
    y_true = []
    y_pred = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()

        y_pred.extend(pred.tolist())
        y_true.extend(yb.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    return acc, mf1, cm, y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--labels", required=True, help="comma-separated labels, e.g. cough,grunt_oink,stress_vocal")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min_delta", type=float, default=5e-4)
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--dur_s", type=float, default=1.0)

    args = ap.parse_args()

    set_seed(args.seed)

    labels = [x.strip().lower() for x in args.labels.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)
    print("[INFO] labels =", labels)

    ds_train = MCTAFDDataset(args.train, labels, sr=args.sr, dur_s=args.dur_s, cache=True)
    ds_val = MCTAFDDataset(args.val, labels, sr=args.sr, dur_s=args.dur_s, cache=True)
    ds_test = MCTAFDDataset(args.test, labels, sr=args.sr, dur_s=args.dur_s, cache=True)

    print("[INFO] train =", len(ds_train))
    print("[INFO] val   =", len(ds_val))
    print("[INFO] test  =", len(ds_test))

    weights, counts = compute_class_weights(ds_train, labels)
    print("[INFO] train label counts =", dict(zip(labels, counts.astype(int).tolist())))
    print("[INFO] class weights =", weights.tolist())

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MCTAFDCRNN(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val = -1.0
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            losses.append(float(loss.item()))

        val_acc, val_mf1, val_cm, _, _ = evaluate(model, dl_val, device, labels)

        scheduler.step(val_mf1)
        current_lr = optim.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_mf1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        improved = val_mf1 > best_val + args.min_delta

        if improved:
            best_val = val_mf1
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": labels,
                    "args": vars(args),
                    "best_val_macro_f1": best_val,
                    "best_epoch": best_epoch,
                },
                args.ckpt,
            )
            print(f"[+] saved best -> {args.ckpt}  best_val={best_val:.4f} epoch={best_epoch}")

        else:
            bad_epochs += 1
            print(f"[EARLYSTOP] no improvement: {bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"[EARLYSTOP] stopped at epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.4f}")
            break

    print("[INFO] loading best checkpoint")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_acc, test_mf1, test_cm, y_true, y_pred = evaluate(model, dl_test, device, labels)

    print("\n[TEST]")
    print(f"ACC={test_acc:.4f}")
    print(f"macro_f1={test_mf1:.4f}")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print("[CONFUSION]")
    print(test_cm)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": [labels[i] for i in y_true],
        "pred_label": [labels[i] for i in y_pred],
    })
    pred_df.to_csv(out_dir / "test_pred.csv", index=False, encoding="utf-8-sig")

    summary = {
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_mf1),
        "confusion": test_cm.tolist(),
        "labels": labels,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] wrote ->", out_dir / "test_pred.csv")
    print("[OK] wrote ->", out_dir / "summary.json")


if __name__ == "__main__":
    main()