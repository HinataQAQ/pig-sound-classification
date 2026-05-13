from pathlib import Path
import pandas as pd
import torch
import argparse


def infer_path_col(df):
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot infer path column. columns={list(df.columns)}")


def norm_paths(csv_path):
    df = pd.read_csv(csv_path)
    col = infer_path_col(df)
    return set(df[col].map(lambda x: str(Path(str(x)).resolve()).lower()))


def print_overlap(name_a, a, name_b, b):
    ov = a & b
    print(f"{name_a}-{name_b} exact overlap = {len(ov)}")
    for p in list(ov)[:20]:
        print("  ", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=r"data\manifests_cough_vs_piggrunt\train.csv")
    ap.add_argument("--val", default=r"data\manifests_cough_vs_piggrunt\val.csv")
    ap.add_argument("--test", default=r"data\manifests_cough_vs_piggrunt\test.csv")
    ap.add_argument("--init_ckpt", default="")
    args = ap.parse_args()

    train = norm_paths(args.train)
    val = norm_paths(args.val)
    test = norm_paths(args.test)

    print("[COUGH_VS_PIGGRUNT SPLIT]")
    print("train =", len(train))
    print("val   =", len(val))
    print("test  =", len(test))

    print_overlap("train", train, "val", val)
    print_overlap("train", train, "test", test)
    print_overlap("val", val, "test", test)

    if args.init_ckpt:
        print("\n[INIT CKPT CHECK]")
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        cfg = ckpt.get("config", {})
        train_manifest = cfg.get("data", {}).get("train_manifest", "")
        print("init_ckpt train_manifest =", train_manifest)

        if train_manifest and Path(train_manifest).exists():
            init_train = norm_paths(train_manifest)
            print("init_train =", len(init_train))
            print_overlap("init_train", init_train, "cough_vs_piggrunt_test", test)
        else:
            print("[WARN] cannot locate init checkpoint train manifest")


if __name__ == "__main__":
    main()