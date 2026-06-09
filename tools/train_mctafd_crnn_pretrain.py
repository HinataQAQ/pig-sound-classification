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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 尽量增强可复现性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    """
    用 soundfile 读取音频，避免部分 wav 在 librosa.load 阶段卡住。
    """
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
    把 [R, T] 插值到 [target_rows, T]。
    用于把 MFCC / Delta / Flux 对齐到 mel bins。
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


def safe_delta(mfcc, order=1):
    """
    防止极短时间帧时 delta 报错。
    """
    t = mfcc.shape[1]
    width = 9

    if t < width:
        width = t if t % 2 == 1 else t - 1

    if width < 3:
        return np.zeros_like(mfcc, dtype=np.float32)

    return librosa.feature.delta(
        mfcc,
        width=width,
        order=order,
        axis=-1,
        mode="nearest",
    ).astype(np.float32)


def make_feature(
    y,
    feature_mode,
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
    feature_mode:
      logmel              -> [1, n_mels, T]
      mel_mfcc            -> [2, n_mels, T]
      mel_mfcc_dyn        -> [3, n_mels, T]
      mel_mfcc_dyn_flux   -> [4, n_mels, T]
      mctafd              -> [5, n_mels, T]

    MCTAFD:
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
    ch_logmel = norm_map(logmel)

    if feature_mode == "logmel":
        return np.stack([ch_logmel], axis=0).astype(np.float32)

    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc).astype(np.float32)
    mfcc_align = align_rows(mfcc, target_rows=n_mels)
    ch_mfcc = norm_map(mfcc_align)

    if feature_mode == "mel_mfcc":
        return np.stack([ch_logmel, ch_mfcc], axis=0).astype(np.float32)

    d1 = safe_delta(mfcc, order=1)
    d2 = safe_delta(mfcc, order=2)

    dyn = np.concatenate([d1, d2], axis=0)
    dyn_align = align_rows(dyn, target_rows=n_mels)
    ch_dyn = norm_map(dyn_align)

    if feature_mode == "mel_mfcc_dyn":
        return np.stack([ch_logmel, ch_mfcc, ch_dyn], axis=0).astype(np.float32)

    diff_t = np.diff(logmel, axis=1)
    flux = np.maximum(diff_t, 0.0)
    flux = np.concatenate([np.zeros((n_mels, 1), dtype=np.float32), flux], axis=1)
    ch_flux = norm_map(flux)

    if feature_mode == "mel_mfcc_dyn_flux":
        return np.stack([ch_logmel, ch_mfcc, ch_dyn, ch_flux], axis=0).astype(np.float32)

    if feature_mode == "mctafd":
        diff_map = np.abs(norm_map(logmel) - norm_map(mfcc_align)).astype(np.float32)
        ch_diff = norm_map(diff_map)

        return np.stack(
            [ch_logmel, ch_mfcc, ch_dyn, ch_flux, ch_diff],
            axis=0,
        ).astype(np.float32)

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def feature_channels(feature_mode):
    mapping = {
        "logmel": 1,
        "mel_mfcc": 2,
        "mel_mfcc_dyn": 3,
        "mel_mfcc_dyn_flux": 4,
        "mctafd": 5,
    }

    if feature_mode not in mapping:
        raise ValueError(f"Unknown feature_mode={feature_mode}")

    return mapping[feature_mode]


class PigVocalFeatureDataset(Dataset):
    def __init__(
        self,
        manifest,
        labels,
        feature_mode,
        sr=32000,
        dur_s=1.0,
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

        self.labels = [x.strip().lower() for x in labels]
        self.label2id = {x: i for i, x in enumerate(self.labels)}

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


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class AblationCRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        fusion_channels=16,
        cnn_channels=(32, 64, 128),
        rnn_hidden=128,
        use_se=True,
        fusion_type="direct",
        aux_gate_init=-4.0,
    ):
        super().__init__()

        if fusion_type not in {"direct", "gated"}:
            raise ValueError(f"Unknown fusion_type={fusion_type}")

        self.in_channels = int(in_channels)
        self.fusion_type = fusion_type

        if self.in_channels > 1 and self.fusion_type == "gated":
            self.aux_gate = nn.Parameter(
                torch.full(
                    (self.in_channels - 1,),
                    float(aux_gate_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.aux_gate = None

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

    def apply_input_gate(self, x):
        """
        x: [B, C, F, T]

        gated fusion:
        - channel 0 = Log-Mel，保持不变
        - channel 1..C-1 = 辅助通道，乘以可学习 gate
        """
        if self.aux_gate is None:
            return x

        if x.size(1) <= 1:
            return x

        logmel = x[:, :1, :, :]
        aux = x[:, 1:, :, :]

        gate = torch.sigmoid(self.aux_gate).view(1, -1, 1, 1)
        aux = aux * gate

        return torch.cat([logmel, aux], dim=1)

    def forward(self, x):
        x = self.apply_input_gate(x)

        x = self.frontend(x)
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        out, _ = self.rnn(x)
        z = out.mean(dim=1)

        logits = self.classifier(z)
        return logits

    def get_aux_gate_values(self):
        if self.aux_gate is None:
            return None
        return torch.sigmoid(self.aux_gate.detach().cpu()).tolist()


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

    ys = []
    ps = []
    probs = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        ys.extend(yb.detach().cpu().numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())
        probs.extend(prob.detach().cpu().numpy().tolist())

    acc = accuracy_score(ys, ps)
    mf1 = f1_score(ys, ps, average="macro")
    cm = confusion_matrix(ys, ps, labels=list(range(len(labels))))

    return acc, mf1, cm, ys, ps, probs


def set_encoder_trainable(model, trainable: bool):
    """
    encoder = frontend + cnn + rnn + aux_gate
    classifier 始终保持可训练。
    """
    for module_name in ["frontend", "cnn", "rnn"]:
        module = getattr(model, module_name, None)
        if module is None:
            continue

        for p in module.parameters():
            p.requires_grad = trainable

    if hasattr(model, "aux_gate") and model.aux_gate is not None:
        model.aux_gate.requires_grad = trainable

    for p in model.classifier.parameters():
        p.requires_grad = True


def load_pretrained_encoder(model, ckpt_path):
    """
    只加载 encoder 部分，跳过 classifier。
    支持：
      torch.save(model.state_dict())
      {"state_dict": state_dict}
      {"model": state_dict}
    """
    ckpt_path = str(ckpt_path)

    print(f"[PRETRAIN] loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model_state = model.state_dict()

    loaded = {}
    skipped_shape = []
    skipped_classifier = []

    for k, v in state.items():
        if k.startswith("classifier."):
            skipped_classifier.append(k)
            continue

        if k in model_state and model_state[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped_shape.append(k)

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)

    print(f"[PRETRAIN] loaded keys      = {len(loaded)}")
    print(f"[PRETRAIN] skipped classifier = {len(skipped_classifier)}")
    print(f"[PRETRAIN] skipped shape/key  = {len(skipped_shape)}")

    if len(skipped_shape) > 0:
        print("[PRETRAIN] skipped examples:")
        for k in skipped_shape[:20]:
            print("  ", k)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--labels", required=True)

    ap.add_argument(
        "--feature_mode",
        required=True,
        choices=[
            "logmel",
            "mel_mfcc",
            "mel_mfcc_dyn",
            "mel_mfcc_dyn_flux",
            "mctafd",
        ],
    )

    ap.add_argument("--no_se", action="store_true")

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
    ap.add_argument("--dur_s", type=float, default=1.0)

    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--fmin", type=float, default=50)
    ap.add_argument("--fmax", type=float, default=8000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=320)
    ap.add_argument("--win_length", type=int, default=800)

    ap.add_argument("--boundary_loss_weight", type=float, default=0.0)
    ap.add_argument("--boundary_warmup_epochs", type=int, default=0)

    ap.add_argument(
        "--fusion_type",
        type=str,
        default="direct",
        choices=["direct", "gated"],
    )
    ap.add_argument("--aux_gate_init", type=float, default=-4.0)

    # 预训练相关参数
    ap.add_argument("--pretrained_ckpt", type=str, default="")
    ap.add_argument("--load_encoder_only", action="store_true")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=0)

    args = ap.parse_args()

    set_seed(args.seed)

    labels = [x.strip().lower() for x in args.labels.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] device =", device)
    print("[INFO] labels =", labels)
    print("[INFO] feature_mode =", args.feature_mode)
    print("[INFO] use_se =", not args.no_se)
    print("[INFO] fusion_type =", args.fusion_type)
    print("[INFO] pretrained_ckpt =", args.pretrained_ckpt)

    ds_train = PigVocalFeatureDataset(
        args.train,
        labels,
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

    ds_val = PigVocalFeatureDataset(
        args.val,
        labels,
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

    ds_test = PigVocalFeatureDataset(
        args.test,
        labels,
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

    print("[INFO] train =", len(ds_train))
    print("[INFO] val   =", len(ds_val))
    print("[INFO] test  =", len(ds_test))

    weights, counts = compute_class_weights(ds_train, labels)

    print("[INFO] train label counts =", dict(zip(labels, counts.astype(int).tolist())))
    print("[INFO] class weights =", weights.tolist())

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

    in_ch = feature_channels(args.feature_mode)

    model = AblationCRNN(
        num_classes=len(labels),
        in_channels=in_ch,
        use_se=not args.no_se,
        fusion_type=args.fusion_type,
        aux_gate_init=args.aux_gate_init,
    ).to(device)

    if args.pretrained_ckpt:
        if args.load_encoder_only:
            load_pretrained_encoder(model, args.pretrained_ckpt)
        else:
            print(f"[PRETRAIN] loading full model: {args.pretrained_ckpt}")
            state = torch.load(args.pretrained_ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)

        model = model.to(device)

    if args.freeze_encoder_epochs > 0:
        print(f"[FREEZE] encoder frozen for first {args.freeze_encoder_epochs} epochs")
        set_encoder_trainable(model, trainable=False)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

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

    feeding_idx = labels.index("feeding") if "feeding" in labels else None
    stress_idx = labels.index("stress_vocal") if "stress_vocal" in labels else None
    can_boundary_loss = feeding_idx is not None and stress_idx is not None

    for epoch in range(1, args.epochs + 1):
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            print("[FREEZE] unfreezing encoder")
            set_encoder_trainable(model, trainable=True)

        model.train()
        losses = []

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            if args.boundary_loss_weight > 0 and can_boundary_loss:
                mask = (yb == feeding_idx) | (yb == stress_idx)

                if mask.any():
                    fs_logits = logits[mask][:, [feeding_idx, stress_idx]]
                    fs_target = (yb[mask] == stress_idx).long()

                    fs_loss = torch.nn.functional.cross_entropy(fs_logits, fs_target)

                    if args.boundary_warmup_epochs > 0:
                        warmup_scale = min(1.0, epoch / float(args.boundary_warmup_epochs))
                    else:
                        warmup_scale = 1.0

                    loss = loss + args.boundary_loss_weight * warmup_scale * fs_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            losses.append(float(loss.item()))

        val_acc, val_mf1, val_cm, _, _, _ = evaluate(model, dl_val, device, labels)

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

            torch.save(model.state_dict(), ckpt_path)

            print(f"[+] saved best -> {ckpt_path}  best_val={best_val:.4f} epoch={epoch}")

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

    test_acc, test_mf1, test_cm, y_true, y_pred, y_prob = evaluate(
        model,
        dl_test,
        device,
        labels,
    )

    print("\n[TEST]")
    print(f"ACC={test_acc:.4f}")
    print(f"macro_f1={test_mf1:.4f}")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )

    print("[CONFUSION]")
    print(test_cm)

    pred_df = pd.DataFrame({
        "y_true": [labels[i] for i in y_true],
        "y_pred": [labels[i] for i in y_pred],
    })

    for i, lab in enumerate(labels):
        pred_df[f"prob_{lab}"] = [float(p[i]) for p in y_prob]

    pred_out = out_dir / "test_pred.csv"
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote -> {pred_out}")

    summary = {
        "feature_mode": args.feature_mode,
        "use_se": not args.no_se,
        "seed": args.seed,

        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "sr": args.sr,
        "dur_s": args.dur_s,

        "boundary_loss_weight": args.boundary_loss_weight,
        "boundary_warmup_epochs": args.boundary_warmup_epochs,

        "fusion_type": args.fusion_type,
        "aux_gate_init": args.aux_gate_init,

        "pretrained_ckpt": args.pretrained_ckpt,
        "load_encoder_only": bool(args.load_encoder_only),
        "freeze_encoder_epochs": int(args.freeze_encoder_epochs),

        "best_val_macro_f1": float(best_val),
        "best_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_mf1),
        "confusion": test_cm.tolist(),
        "labels": labels,
    }

    if hasattr(model, "get_aux_gate_values"):
        summary["aux_gate"] = model.get_aux_gate_values()

    summary_out = out_dir / "summary.json"

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote -> {summary_out}")


if __name__ == "__main__":
    main()