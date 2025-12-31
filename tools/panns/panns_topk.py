# tools/panns_topk.py
import argparse
import numpy as np

from panns_inference import AudioTagging
from panns_inference.config import labels

import librosa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--sr", type=int, default=32000)   # PANNs 通常用 32k
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    y, _ = librosa.load(args.wav, sr=args.sr, mono=True)
    need = int(args.seconds * args.sr)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    else:
        y = y[:need]
    y = y.astype(np.float32)

    at = AudioTagging(checkpoint_path=None, device="cpu")
    (clipwise_output, embedding) = at.inference(y[None, :])

    probs = clipwise_output[0]
    idx = np.argsort(probs)[::-1][:args.topk]

    print("\n=== TOPK LABELS ===")
    for i in idx:
        print(f"{labels[i]:30s}  {probs[i]:.4f}")

    # 额外：把 cough/sneeze 相关的都单独打印出来（方便判断）
    key_words = ["cough", "sneeze", "throat", "sniff", "breath", "respir"]
    print("\n=== COUGH-RELATED PROBS (substring match) ===")
    for i, name in enumerate(labels):
        low = name.lower()
        if any(k in low for k in key_words):
            print(f"{i:3d}  {name:30s}  {probs[i]:.4f}")

    print("\n[OK] embedding dim =", embedding.shape)

if __name__ == "__main__":
    main()
