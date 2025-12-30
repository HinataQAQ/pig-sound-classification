import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml

from torch.utils.data import Dataset, DataLoader

from .utils.audio_io import load_wav, fix_length
from .preprocessing.denoise import spectral_subtraction


def auto_device(flag: str) -> str:
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_logmel(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    # 1s@32000Hz + n_fft=1024, hop=320, center=False -> 97 frames（你前面测出来就是 97）
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 800,
) -> np.ndarray:
    """
    返回 (T, F) 的 log-mel 特征（float32）
    """
    import librosa

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
        center=False,
    )
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)  # 大致 [-80, 0]
    feat = logmel.T.astype(np.float32)  # (T, F)
    return feat


class LogMelDataset(Dataset):
    def __init__(self, manifest_csv: str, labels, sr: int, segment_seconds: float, do_denoise: bool, train: bool):
        self.df = pd.read_csv(manifest_csv)
        self.labels = [str(x).strip().lower() for x in labels]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.sr = int(sr)
        self.seg_len = int(round(self.sr * float(segment_seconds)))
        self.do_denoise = bool(do_denoise)
        self.train = bool(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        raw_lab = str(row["label"])
        lab = raw_lab.strip().lower()
        if lab not in self.label2id:
            raise KeyError(f"Label '{raw_lab}' not in labels {self.labels}")
        y_id = self.label2id[lab]

        y, sr = load_wav(path, sr=self.sr, mono=True)
        if self.do_denoise:
            y = spectral_subtraction(y, sr)

        # 统一 1s（训练/验证/测试都一样长度，验证/测试取中间片段更稳）
        if len(y) < self.seg_len:
            y = fix_length(y, self.seg_len)
        elif len(y) > self.seg_len:
            start = 0 if self.train else max(0, (len(y) - self.seg_len) // 2)
            y = y[start:start + self.seg_len]

        feat = compute_logmel(y, sr=self.sr, n_mels=64, fmin=50, fmax=8000)  # (T,F)

        # 建议：做一个简单的 utterance-level 标准化（和你 MFCC/GFCC 那套一致）
        mu = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mu) / std

        x = torch.from_numpy(feat)          # (T,F)
        ylab = torch.tensor(y_id).long()
        return x, ylab, path


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def confusion_mat(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def build_crnn_model(num_classes: int, n_mels: int, cfg_model: dict):
    """
    尽量自动适配你 src/models/crnn.py 里的类名和 init 参数
    """
    import inspect
    from .models import crnn as crnn_mod

    # 找候选 nn.Module 类
    cands = []
    for name, obj in crnn_mod.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            cands.append((name, obj))

    # 优先挑名字里带 CRNN 的
    cands.sort(key=lambda x: ("crnn" not in x[0].lower(), x[0].lower()))
    if not cands:
        raise RuntimeError("No nn.Module class found in src/models/crnn.py")

    last_err = None
    for name, cls in cands:
        try:
            sig = inspect.signature(cls.__init__)
            kwargs = {}

            # 常见参数名兼容
            for k in ["num_classes", "classes_num", "n_classes", "n_class"]:
                if k in sig.parameters:
                    kwargs[k] = num_classes
                    break

            for k in ["n_mels", "n_mel", "mel_bins", "n_bins"]:
                if k in sig.parameters:
                    kwargs[k] = n_mels
                    break

            # 如果你的类还支持 dropout 等，就顺便塞进去
            if "dropout" in sig.parameters and "dropout" in cfg_model:
                kwargs["dropout"] = cfg_model["dropout"]

            model = cls(**kwargs)
            return model
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to instantiate CRNN model from src/models/crnn.py, last_err={last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--config", default=None, type=str, help="yaml config path (optional if ckpt has config)")
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--device", default="auto", type=str)
    ap.add_argument("--out", default="eval_crnn_predictions.csv", type=str)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # 你之前打印过 ckpt keys: ['state_dict','labels','config']
    cfg = ckpt.get("config", None)
    if cfg is None:
        if not args.config:
            raise KeyError("checkpoint has no 'config' and you didn't pass --config")
        cfg = load_yaml(args.config)

    labels = ckpt.get("labels", cfg["data"]["labels"])
    labels = [str(x).strip().lower() for x in labels]

    device = auto_device(args.device)
    print(f"[INFO] device = {device}")

    sr = int(cfg["data"].get("sample_rate", 32000))
    seg_s = float(cfg["data"].get("segment_seconds", 1.0))
    do_denoise = bool(cfg.get("preprocess", {}).get("do_denoise", False))
    model_cfg = cfg.get("model", {})

    ds = LogMelDataset(
        manifest_csv=args.manifest,
        labels=labels,
        sr=sr,
        segment_seconds=seg_s,
        do_denoise=do_denoise,
        train=False,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_crnn_model(num_classes=len(labels), n_mels=64, cfg_model=model_cfg)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict not strict:")
        print("  missing   =", missing[:20], ("..." if len(missing) > 20 else ""))
        print("  unexpected=", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    model.to(device)
    model.eval()

    all_true, all_pred, all_prob = [], [], []
    all_path = []

    with torch.inference_mode():
        for x, y, p in dl:
            x = x.to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            pred = prob.argmax(axis=-1)

            all_true.append(y.numpy())
            all_pred.append(pred)
            all_prob.append(prob)
            all_path += list(p)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    prob = np.concatenate(all_prob, axis=0)

    acc = float((y_true == y_pred).mean())
    mf1 = macro_f1(y_true, y_pred, num_classes=len(labels))
    cm = confusion_mat(y_true, y_pred, num_classes=len(labels))

    print(f"[OK] ACC={acc:.4f}, Macro-F1={mf1:.4f}")
    print("[OK] confusion (rows=true, cols=pred):")
    print(cm)

    # 保存预测
    id2label = {i: l for i, l in enumerate(labels)}
    out_df = pd.DataFrame({
        "filepath": all_path,
        "y_true": [id2label[int(i)] for i in y_true],
        "y_pred": [id2label[int(i)] for i in y_pred],
    })
    # 概率列：prob_<label>
    for i, lab in enumerate(labels):
        out_df[f"prob_{lab}"] = prob[:, i].astype(np.float32)

    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] wrote -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
