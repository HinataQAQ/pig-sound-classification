from pathlib import Path
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from train_mctafd_crnn_ablation import (
    load_audio_soundfile,
    make_feature,
    feature_channels,
    SEBlock,
    IdentityBlock,
)


ROOT = Path(__file__).resolve().parents[1]


MAIN_LABELS_DEFAULT = [
    "cough",
    "calm_grunt",
    "feeding",
    "stress_vocal",
]

AUX_LABELS_DEFAULT = [
    "dry_cough",
    "abdominal_cough",
    "calm_grunt",
    "feeding",
    "frightened_stress",
    "anxious_stress",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_path_col(df):
    for c in ["path", "filepath", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column from columns={list(df.columns)}")


def infer_label_col(df):
    for c in ["label", "target", "y"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer label column from columns={list(df.columns)}")


def resolve_path(p):
    q = Path(str(p).replace("\\", "/"))
    if q.is_absolute():
        return q
    return ROOT / q


def normalize_label(x):
    return str(x).strip().lower()


def infer_subtype_from_row(row, path_col, label_col):
    """
    优先使用 manifest 的 subtype 列。
    如果没有 subtype，则从路径和主标签推断。
    """
    main = normalize_label(row[label_col])

    if "subtype" in row.index:
        st = normalize_label(row["subtype"])
        if st not in {"", "nan", "none"}:
            return st

    path = str(row[path_col]).replace("\\", "/").lower()

    if main == "cough":
        if "abdominal_cough" in path:
            return "abdominal_cough"
        if "dry_cough" in path:
            return "dry_cough"
        return "dry_cough"

    if main == "stress_vocal":
        if "frightened_stress" in path:
            return "frightened_stress"
        if "anxious_stress" in path:
            return "anxious_stress"
        return "frightened_stress"

    if main == "calm_grunt":
        return "calm_grunt"

    if main == "feeding":
        return "feeding"

    raise RuntimeError(f"Cannot infer subtype for main label={main}, path={path}")


class HierPigVocalDataset(Dataset):
    def __init__(
        self,
        manifest,
        main_labels,
        aux_labels,
        feature_mode,
        sr=32000,
        dur_s=2.0,
        cache=True,
        n_mels=64,
        n_mfcc=20,
        n_fft=1024,
        hop_length=320,
        win_length=800,
        fmin=50,
        fmax=8000,
    ):
        self.df = pd.read_csv(manifest)

        self.path_col = infer_path_col(self.df)
        self.label_col = infer_label_col(self.df)

        self.main_labels = [x.strip().lower() for x in main_labels]
        self.aux_labels = [x.strip().lower() for x in aux_labels]

        self.main2id = {x: i for i, x in enumerate(self.main_labels)}
        self.aux2id = {x: i for i, x in enumerate(self.aux_labels)}

        self.feature_mode = feature_mode
        self.sr = sr
        self.dur_s = dur_s
        self.cache = cache
        self._cache = {}

        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax

        self.df[self.label_col] = self.df[self.label_col].astype(str).str.strip().str.lower()

        keep = self.df[self.label_col].isin(self.main2id.keys())
        bad = self.df[~keep]

        if len(bad) > 0:
            print("[WARN] dropping rows with unknown main labels:")
            print(bad[self.label_col].value_counts())

        self.df = self.df[keep].reset_index(drop=True)

        # 预先推断 subtype，避免训练时反复推断。
        aux_list = []
        keep_aux = []

        for _, r in self.df.iterrows():
            st = infer_subtype_from_row(r, self.path_col, self.label_col)
            aux_list.append(st)
            keep_aux.append(st in self.aux2id)

        self.df["__aux_label"] = aux_list
        self.df = self.df[keep_aux].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"No valid rows after subtype filtering: {manifest}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]

        path = resolve_path(row[self.path_col])
        main_lab = normalize_label(row[self.label_col])
        aux_lab = normalize_label(row["__aux_label"])

        y, sr = load_audio_soundfile(path, sr=self.sr, dur_s=self.dur_s)

        x = make_feature(
            y,
            feature_mode=self.feature_mode,
            sr=sr,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        x = torch.from_numpy(x).float()
        y_main = torch.tensor(self.main2id[main_lab], dtype=torch.long)
        y_aux = torch.tensor(self.aux2id[aux_lab], dtype=torch.long)

        item = (x, y_main, y_aux)

        if self.cache:
            self._cache[idx] = item

        return item


class HierCRNN(nn.Module):
    def __init__(
        self,
        num_main_classes,
        num_aux_classes,
        in_channels,
        fusion_channels=16,
        cnn_channels=(32, 64, 128),
        rnn_hidden=128,
        use_se=True,
        rnn_type="gru",
        pooling_type="mean",
    ):
        super().__init__()

        if rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unknown rnn_type={rnn_type}")

        if pooling_type not in {"mean", "attn"}:
            raise ValueError(f"Unknown pooling_type={pooling_type}")

        self.rnn_type = rnn_type
        self.pooling_type = pooling_type

        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            SEBlock(fusion_channels, reduction=4) if use_se else IdentityBlock(),
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

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cnn_channels[-1],
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=cnn_channels[-1],
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        if pooling_type == "attn":
            self.attn = nn.Sequential(
                nn.Linear(rnn_hidden * 2, rnn_hidden),
                nn.Tanh(),
                nn.Linear(rnn_hidden, 1),
            )
        else:
            self.attn = None

        self.main_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden * 2, num_main_classes),
        )

        self.aux_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden * 2, num_aux_classes),
        )

    def encode(self, x):
        x = self.frontend(x)
        x = self.cnn(x)          # [B, C, F, T]
        x = x.mean(dim=2)        # [B, C, T]
        x = x.permute(0, 2, 1)   # [B, T, C]

        out, _ = self.rnn(x)

        if self.pooling_type == "mean":
            z = out.mean(dim=1)

        elif self.pooling_type == "attn":
            score = self.attn(out).squeeze(-1)
            weight = torch.softmax(score, dim=1)
            z = torch.sum(out * weight.unsqueeze(-1), dim=1)

        else:
            raise ValueError(f"Unknown pooling_type={self.pooling_type}")

        return z

    def forward(self, x):
        z = self.encode(x)
        main_logits = self.main_head(z)
        aux_logits = self.aux_head(z)
        return main_logits, aux_logits


def compute_weights_from_dataset(ds, kind="main"):
    if kind == "main":
        labels = ds.main_labels
        col = ds.label_col
        label2id = ds.main2id
        vals = [normalize_label(x) for x in ds.df[col].tolist()]
    elif kind == "aux":
        labels = ds.aux_labels
        label2id = ds.aux2id
        vals = [normalize_label(x) for x in ds.df["__aux_label"].tolist()]
    else:
        raise ValueError(kind)

    counts = np.zeros(len(labels), dtype=np.float32)

    for v in vals:
        counts[label2id[v]] += 1

    weights = counts.sum() / (len(labels) * np.maximum(counts, 1.0))

    return torch.tensor(weights, dtype=torch.float32), counts


@torch.no_grad()
def evaluate(model, loader, device, main_labels, aux_labels):
    model.eval()

    y_main_true = []
    y_main_pred = []

    y_aux_true = []
    y_aux_pred = []

    probs = []

    for xb, y_main, y_aux in loader:
        xb = xb.to(device)
        y_main = y_main.to(device)
        y_aux = y_aux.to(device)

        main_logits, aux_logits = model(xb)

        main_prob = torch.softmax(main_logits, dim=1)
        main_pred = torch.argmax(main_prob, dim=1)

        aux_pred = torch.argmax(aux_logits, dim=1)

        y_main_true.extend(y_main.detach().cpu().numpy().tolist())
        y_main_pred.extend(main_pred.detach().cpu().numpy().tolist())

        y_aux_true.extend(y_aux.detach().cpu().numpy().tolist())
        y_aux_pred.extend(aux_pred.detach().cpu().numpy().tolist())

        probs.extend(main_prob.detach().cpu().numpy().tolist())

    main_acc = accuracy_score(y_main_true, y_main_pred)
    main_mf1 = f1_score(
        y_main_true,
        y_main_pred,
        labels=list(range(len(main_labels))),
        average="macro",
        zero_division=0,
    )
    main_cm = confusion_matrix(
        y_main_true,
        y_main_pred,
        labels=list(range(len(main_labels))),
    )

    aux_acc = accuracy_score(y_aux_true, y_aux_pred)
    aux_mf1 = f1_score(
        y_aux_true,
        y_aux_pred,
        labels=list(range(len(aux_labels))),
        average="macro",
        zero_division=0,
    )

    return {
        "main_acc": main_acc,
        "main_macro_f1": main_mf1,
        "main_cm": main_cm,
        "y_main_true": y_main_true,
        "y_main_pred": y_main_pred,
        "y_aux_true": y_aux_true,
        "y_aux_pred": y_aux_pred,
        "aux_acc": aux_acc,
        "aux_macro_f1": aux_mf1,
        "main_probs": probs,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)

    ap.add_argument("--labels", default="cough,calm_grunt,feeding,stress_vocal")
    ap.add_argument(
        "--aux_labels",
        default="dry_cough,abdominal_cough,calm_grunt,feeding,frightened_stress,anxious_stress",
    )

    ap.add_argument(
        "--feature_mode",
        default="logmel",
        choices=[
            "logmel",
            "logmel_hpf",
            "logmel_sg",
            "logmel_pcen",
            "logmel_sg_pcen",
            "logmel_dual",
            "logmel_dual_pcen",
            "mel_mfcc",
            "mel_mfcc_dyn",
            "mel_mfcc_dyn_flux",
            "mctafd",
        ],
    )

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--min_delta", type=float, default=1e-6)

    ap.add_argument("--seed", type=int, default=3407)

    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--dur_s", type=float, default=2.0)

    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--fmin", type=float, default=50)
    ap.add_argument("--fmax", type=float, default=8000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=320)
    ap.add_argument("--win_length", type=int, default=800)

    ap.add_argument("--no_se", action="store_true")
    ap.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "attn"])

    ap.add_argument("--hier_aux_weight", type=float, default=0.5)

    args = ap.parse_args()

    set_seed(args.seed)

    main_labels = [x.strip().lower() for x in args.labels.split(",") if x.strip()]
    aux_labels = [x.strip().lower() for x in args.aux_labels.split(",") if x.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] device =", device)
    print("[INFO] main_labels =", main_labels)
    print("[INFO] aux_labels =", aux_labels)
    print("[INFO] feature_mode =", args.feature_mode)
    print("[INFO] dur_s =", args.dur_s)
    print("[INFO] hier_aux_weight =", args.hier_aux_weight)

    common_ds_kwargs = dict(
        main_labels=main_labels,
        aux_labels=aux_labels,
        feature_mode=args.feature_mode,
        sr=args.sr,
        dur_s=args.dur_s,
        cache=True,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    ds_train = HierPigVocalDataset(args.train, **common_ds_kwargs)
    ds_val = HierPigVocalDataset(args.val, **common_ds_kwargs)
    ds_test = HierPigVocalDataset(args.test, **common_ds_kwargs)

    print("[INFO] train =", len(ds_train))
    print("[INFO] val   =", len(ds_val))
    print("[INFO] test  =", len(ds_test))

    main_weights, main_counts = compute_weights_from_dataset(ds_train, kind="main")
    aux_weights, aux_counts = compute_weights_from_dataset(ds_train, kind="aux")

    print("[INFO] main train counts =", dict(zip(main_labels, main_counts.astype(int).tolist())))
    print("[INFO] aux train counts  =", dict(zip(aux_labels, aux_counts.astype(int).tolist())))
    print("[INFO] main weights =", main_weights.tolist())
    print("[INFO] aux weights  =", aux_weights.tolist())

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = HierCRNN(
        num_main_classes=len(main_labels),
        num_aux_classes=len(aux_labels),
        in_channels=feature_channels(args.feature_mode),
        use_se=not args.no_se,
        rnn_type=args.rnn_type,
        pooling_type=args.pooling_type,
    ).to(device)

    main_criterion = nn.CrossEntropyLoss(weight=main_weights.to(device))
    aux_criterion = nn.CrossEntropyLoss(weight=aux_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        main_losses = []
        aux_losses = []

        for xb, y_main, y_aux in dl_train:
            xb = xb.to(device)
            y_main = y_main.to(device)
            y_aux = y_aux.to(device)

            optimizer.zero_grad()

            main_logits, aux_logits = model(xb)

            main_loss = main_criterion(main_logits, y_main)
            aux_loss = aux_criterion(aux_logits, y_aux)

            loss = main_loss + args.hier_aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))
            main_losses.append(float(main_loss.item()))
            aux_losses.append(float(aux_loss.item()))

        val = evaluate(model, dl_val, device, main_labels, aux_labels)
        val_mf1 = val["main_macro_f1"]

        scheduler.step(val_mf1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"main_loss={np.mean(main_losses):.4f} | "
            f"aux_loss={np.mean(aux_losses):.4f} | "
            f"val_acc={val['main_acc']:.4f} | "
            f"val_macro_f1={val_mf1:.4f} | "
            f"val_aux_f1={val['aux_macro_f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        improved = val_mf1 > best_val + args.min_delta

        if improved:
            best_val = val_mf1
            best_epoch = epoch
            bad_epochs = 0

            torch.save(model.state_dict(), ckpt_path)
            print(f"[+] saved best -> {ckpt_path} best_val={best_val:.4f} epoch={epoch}")

        else:
            bad_epochs += 1
            print(f"[EARLYSTOP] no improvement: {bad_epochs}/{args.patience}")

            if bad_epochs >= args.patience:
                print(
                    f"[EARLYSTOP] stopped at epoch={epoch}, "
                    f"best_epoch={best_epoch}, best_val={best_val:.4f}"
                )
                break

    print("[INFO] loading best checkpoint")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    test = evaluate(model, dl_test, device, main_labels, aux_labels)

    test_acc = test["main_acc"]
    test_mf1 = test["main_macro_f1"]
    test_cm = test["main_cm"]

    print("\n[TEST MAIN]")
    print(f"ACC={test_acc:.4f}")
    print(f"macro_f1={test_mf1:.4f}")
    print(
        classification_report(
            test["y_main_true"],
            test["y_main_pred"],
            target_names=main_labels,
            digits=4,
            zero_division=0,
        )
    )

    print("[CONFUSION MAIN]")
    print(test_cm)

    print("\n[TEST AUX]")
    print(f"aux_acc={test['aux_acc']:.4f}")
    print(f"aux_macro_f1={test['aux_macro_f1']:.4f}")

    pred_df = pd.DataFrame({
        "y_true": [main_labels[i] for i in test["y_main_true"]],
        "y_pred": [main_labels[i] for i in test["y_main_pred"]],
        "aux_true": [aux_labels[i] for i in test["y_aux_true"]],
        "aux_pred": [aux_labels[i] for i in test["y_aux_pred"]],
    })

    for i, lab in enumerate(main_labels):
        pred_df[f"prob_{lab}"] = [float(p[i]) for p in test["main_probs"]]

    pred_out = out_dir / "test_pred.csv"
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote -> {pred_out}")

    summary = {
        "model_family": "hier_longcontext_crnn",
        "feature_mode": args.feature_mode,
        "use_se": not args.no_se,
        "seed": args.seed,

        "dur_s": args.dur_s,
        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "sr": args.sr,

        "rnn_type": args.rnn_type,
        "pooling_type": args.pooling_type,

        "hier_aux": True,
        "hier_aux_weight": args.hier_aux_weight,
        "main_labels": main_labels,
        "aux_labels": aux_labels,

        "best_val_macro_f1": float(best_val),
        "best_epoch": int(best_epoch),

        "test_acc": float(test_acc),
        "test_macro_f1": float(test_mf1),
        "test_aux_acc": float(test["aux_acc"]),
        "test_aux_macro_f1": float(test["aux_macro_f1"]),
        "confusion": test_cm.tolist(),
    }

    summary_out = out_dir / "summary.json"

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote -> {summary_out}")


if __name__ == "__main__":
    main()