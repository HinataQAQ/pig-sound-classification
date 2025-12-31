# tools/test_panns_one.py
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True, help="path to a wav file")
    ap.add_argument("--seconds", type=float, default=1.0, help="target seconds to pad/crop")
    ap.add_argument("--sr", type=int, default=32000, help="PANNs expects 32000 in examples")
    args = ap.parse_args()

    try:
        import librosa
    except Exception as e:
        print("[ERR] librosa not available:", repr(e))
        print("请先安装：python -m pip install librosa soundfile")
        return

    from panns_inference import AudioTagging

    # load & resample to 32000 like official example
    audio, _ = librosa.load(args.wav, sr=args.sr, mono=True)

    # pad / crop to fixed length
    target_len = int(args.seconds * args.sr)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    audio = audio[None, :]  # (batch, samples)

    at = AudioTagging(checkpoint_path=None, device="cpu")
    clipwise_output, embedding = at.inference(audio)

    print("[OK] clipwise_output shape =", clipwise_output.shape)
    print("[OK] embedding shape       =", embedding.shape)
    print("[OK] embedding first 8 dims =", embedding[0, :8])

if __name__ == "__main__":
    main()
