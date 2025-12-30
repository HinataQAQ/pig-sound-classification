# tools/extract_panns_embeddings.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sr", type=int, default=32000, help="PANNs example uses 32000")
    ap.add_argument("--seconds", type=float, default=None, help="pad/crop length; default from config data.segment_seconds")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 means full; else only first N rows for quick test")
    return ap.parse_args()

def fix_len(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def main():
    args = parse()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(open(root / args.config, "r", encoding="utf-8"))

    # manifests from config (this is why we chose dataset A: config points to cough_binary)
    mkey = {"train": "train_manifest", "val": "val_manifest", "test": "test_manifest"}[args.split]
    manifest_path = Path(cfg["data"][mkey])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    labels_cfg = [str(x).strip().lower() for x in cfg["data"]["labels"]]
    label2id = {l:i for i,l in enumerate(labels_cfg)}

    seconds = args.seconds if args.seconds is not None else float(cfg["data"]["segment_seconds"])
    target_len = int(args.sr * seconds)

    df = pd.read_csv(manifest_path)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    print(f"[INFO] split={args.split} rows={len(df)} manifest={manifest_path}")
    print("[INFO] labels=", labels_cfg)

    # audio loader
    import librosa
    from panns_inference import AudioTagging

    at = AudioTagging(checkpoint_path=None, device="cpu")

    X_list = []
    y_list = []
    path_list = []

    batch_audio = []
    batch_y = []
    batch_paths = []

    def flush():
        nonlocal batch_audio, batch_y, batch_paths
        if not batch_audio:
            return
        audio = np.stack(batch_audio, axis=0).astype(np.float32)  # (B, samples)
        _, emb = at.inference(audio)  # emb: (B, 2048) for CNN14
        X_list.append(emb.astype(np.float32))
        y_list.append(np.array(batch_y, dtype=np.int64))
        path_list.extend(batch_paths)
        batch_audio, batch_y, batch_paths = [], [], []

    for i, row in df.iterrows():
        p = str(row["filepath"])
        p_norm = p.replace("\\", "/")
        wav_path = Path(p_norm)
        if not wav_path.is_absolute():
            wav_path = root / wav_path
        if not wav_path.exists():
            raise FileNotFoundError(f"wav not found: {wav_path}")

        raw_lab = str(row["label"]).strip().lower()
        if raw_lab not in label2id:
            raise KeyError(f"label '{raw_lab}' not in config labels {labels_cfg}")
        yid = label2id[raw_lab]

        y, _ = librosa.load(str(wav_path), sr=args.sr, mono=True)  # resample to sr
        y = fix_len(y, target_len)

        batch_audio.append(y)
        batch_y.append(yid)
        batch_paths.append(p_norm)

        if len(batch_audio) >= args.batch_size:
            flush()

        if (i + 1) % 200 == 0:
            print(f"[INFO] processed {i+1}/{len(df)}")

    flush()

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        paths=np.array(path_list, dtype=object),
        labels=np.array(labels_cfg, dtype=object),
        sr=args.sr,
        seconds=seconds,
    )

    print(f"[OK] saved -> {out_path}")
    print("[OK] X shape =", X.shape, "y shape =", y.shape)
    # class balance
    uniq, cnt = np.unique(y, return_counts=True)
    print("[OK] class counts:", {labels_cfg[int(k)]: int(v) for k, v in zip(uniq, cnt)})

if __name__ == "__main__":
    main()
