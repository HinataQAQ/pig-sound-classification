from pathlib import Path
import re
import pandas as pd
import argparse


def infer_path_col(df):
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot find path column: {list(df.columns)}")


def source_id_from_path(p):
    stem = Path(str(p)).stem

    # processed slices:
    # aswine_ala_e_2_2020-09-30_05-58-11_488_t0000053000
    m = re.match(r"(aswine_.+?)_t\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # round4 event clips:
    # ALA_E_...__rank...
    if "__rank" in stem:
        return stem.split("__rank")[0].lower()

    # old rank clips:
    m = re.match(r"(.+?)_rank\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return stem.lower()


def load_sources(path):
    df = pd.read_csv(path)
    col = infer_path_col(df)
    sources = set(df[col].map(source_id_from_path))
    paths = set(df[col].map(lambda x: str(Path(str(x)).resolve()).lower()))
    return sources, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()

    tr_src, tr_paths = load_sources(args.train)
    va_src, va_paths = load_sources(args.val)
    te_src, te_paths = load_sources(args.test)

    print("[EXACT PATH OVERLAP]")
    print("train-val :", len(tr_paths & va_paths))
    print("train-test:", len(tr_paths & te_paths))
    print("val-test  :", len(va_paths & te_paths))

    print("\n[SOURCE ID OVERLAP]")
    print("train-val :", len(tr_src & va_src))
    print("train-test:", len(tr_src & te_src))
    print("val-test  :", len(va_src & te_src))

    print("\n[train-test overlapping source examples]")
    for s in sorted(list(tr_src & te_src))[:50]:
        print(s)


if __name__ == "__main__":
    main()