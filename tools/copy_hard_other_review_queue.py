from pathlib import Path
import argparse
import shutil
import pandas as pd


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Cannot find columns {candidates}. Existing columns={list(df.columns)}")
    return None


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topn", type=int, default=300)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print("[INFO] rows in csv =", len(df))
    print("[INFO] columns =", list(df.columns))

    path_col = find_col(df, ["filepath", "path", "clip_path", "audio_path", "wav_path"])
    prob_col = find_col(df, ["prob_pos", "prob_cough", "p_cough", "cough_prob", "score_cough", "prob"], required=False)
    source_col = find_col(df, ["source"], required=False)

    # 只在 prob 列真实有数值时排序；如果 prob 全空，就按 CSV 原顺序复制
    use_prob = False
    if prob_col is not None:
        df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
        if df[prob_col].notna().sum() > 0:
            use_prob = True
            df = df.sort_values(prob_col, ascending=False)
        else:
            print(f"[WARN] column {prob_col} exists but all values are empty. Copying by CSV order.")

    df = df.head(args.topn).copy()

    rows = []
    copied = 0
    missing = 0

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        src = Path(str(row[path_col]).strip().strip('"'))

        if not src.exists():
            missing += 1
            rows.append({
                "rank": rank,
                "original_path": str(src),
                "review_path": "",
                "status": "missing",
            })
            continue

        if use_prob:
            prob = float(row[prob_col])
            prob_part = f"p{prob:.3f}"
        else:
            prob = ""
            prob_part = "pNA"

        if source_col is not None and pd.notna(row[source_col]) and str(row[source_col]).strip():
            source = str(row[source_col]).strip()
        else:
            # 从路径里猜一下来源
            path_lower = str(src).lower()
            if "soundwel" in path_lower:
                source = "soundwel"
            elif "wu" in path_lower:
                source = "wu"
            else:
                source = "unknown"

        stem_short = safe_name(src.stem)
        if len(stem_short) > 60:
            stem_short = stem_short[:60]

        new_name = f"rank{rank:03d}_{prob_part}_{safe_name(source)}_{stem_short}{src.suffix.lower()}"
        dst = out_dir / new_name

        shutil.copy2(src, dst)

        copied += 1
        rows.append({
            "rank": rank,
            "prob_cough": prob,
            "source": source,
            "original_path": str(src.resolve()),
            "review_path": str(dst.resolve()),
            "status": "copied",
        })

    index_path = out_dir / "review_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False, encoding="utf-8-sig")

    print("[OK] copied =", copied)
    print("[INFO] missing =", missing)
    print("[OK] out_dir ->", out_dir)
    print("[OK] index  ->", index_path)


if __name__ == "__main__":
    main()