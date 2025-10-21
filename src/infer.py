import argparse, yaml, torch
import numpy as np
from .utils.audio_io import load_wav, fix_length
from .preprocessing.denoise import spectral_subtraction
from .features.mfcc import compute_mfcc, add_deltas as mfcc_delta
from .features.gfcc import compute_gfcc, add_deltas as gfcc_delta
from .models.bilstm import BiLSTMClassifier

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--wav", type=str, required=True)
    ap.add_argument("--config", type=str, default="config/config.yaml")
    return ap.parse_args()

def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ckpt = torch.load(args.model, map_location="cpu")
    labels = ckpt["labels"]
    sr = cfg["data"]["sample_rate"]
    seg_len = int(sr * cfg["data"]["segment_seconds"])

    y, _ = load_wav(args.wav, sr=sr, mono=True)
    y = spectral_subtraction(y, sr)
    y = fix_length(y, seg_len)

    mfcc = compute_mfcc(y, sr=sr)
    mfcc = mfcc_delta(mfcc, cfg["features"]["include_delta"], cfg["features"]["include_delta_delta"])
    gfcc = compute_gfcc(y, sr=sr)
    gfcc = gfcc_delta(gfcc, cfg["features"]["include_delta"], cfg["features"]["include_delta_delta"])
    feat = np.concatenate([mfcc, gfcc], axis=1).astype(np.float32)
    mu = feat.mean(axis=0, keepdims=True); std = feat.std(axis=0, keepdims=True) + 1e-6
    feat = (feat - mu) / std
    x = torch.from_numpy(feat)[None, :, :]  # (1, T, F)

    model = BiLSTMClassifier(input_size=ckpt["input_size"], num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)[0].numpy()
    pred = labels[int(np.argmax(prob))]
    print("Prediction:", pred)
    for i, p in enumerate(prob):
        print(f"{labels[i]:12s}: {p:.3f}")

if __name__ == "__main__":
    main()
