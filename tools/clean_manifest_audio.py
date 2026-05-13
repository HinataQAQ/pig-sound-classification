from pathlib import Path
import sys
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.audio_io import load_wav


def infer_path_col(df: pd.DataFrame) -> str:
    for c in ["filepath", "path", "audio_path", "wav_path"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find path column in manifest. columns={list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_good", required=True)
    ap.add_argument("--out_bad", required=True)
    ap.add_argument("--sr", type=int, default=32000)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    df = pd.read_csv(manifest)
    path_col = infer_path_col(df)

    good_rows = []
    bad_rows = []

    for i, row in df.iterrows():
        p = str(row[path_col]).strip().strip('"')
        try:
            if not Path(p).exists():
                raise FileNotFoundError(p)

            y, sr = load_wav(p, sr=args.sr, mono=True)

            if y is None or len(y) == 0:
                raise RuntimeError("empty audio")

            good_rows.append(row.to_dict())

        except Exception as e:
            bad = row.to_dict()
            bad["error"] = repr(e)
            bad_rows.append(bad)
            print(f"[BAD] idx={i} path={p} error={repr(e)}")

    good_df = pd.DataFrame(good_rows)
    bad_df = pd.DataFrame(bad_rows)

    Path(args.out_good).parent.mkdir(parents=True, exist_ok=True)
    good_df.to_csv(args.out_good, index=False, encoding="utf-8-sig")
    bad_df.to_csv(args.out_bad, index=False, encoding="utf-8-sig")

    print(f"[OK] good rows = {len(good_df)} -> {args.out_good}")
    print(f"[OK] bad rows  = {len(bad_df)} -> {args.out_bad}")


if __name__ == "__main__":
    main()