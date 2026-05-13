from pathlib import Path
import argparse
import pandas as pd
import librosa
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Cannot find any of {candidates}. Existing columns={list(df.columns)}")
    return None


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s


def build_audio_map(audio_dir: Path):
    audio_map = {}
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            audio_map[p.stem] = p
    return audio_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_dir", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--topk_per_file", type=int, default=20)
    ap.add_argument("--min_score", type=float, default=0.65)
    ap.add_argument("--pad_s", type=float, default=0.20)
    ap.add_argument("--min_dur_s", type=float, default=0.10)
    ap.add_argument("--max_dur_s", type=float, default=3.00)

    args = ap.parse_args()

    events_dir = Path(args.events_dir)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_map = build_audio_map(audio_dir)
    records = []

    event_files = sorted(events_dir.glob("*_events.csv"))
    if not event_files:
        raise RuntimeError(f"No *_events.csv found in {events_dir}")

    for event_csv in event_files:
        stem = event_csv.name.replace("_events.csv", "")
        if stem not in audio_map:
            print(f"[WARN] cannot find source wav for {stem}")
            continue

        wav_path = audio_map[stem]
        df = pd.read_csv(event_csv)

        start_col = find_col(df, ["start_s", "start", "onset"])
        end_col = find_col(df, ["end_s", "end", "offset"])
        score_col = find_col(df, ["max_prob", "score", "prob", "confidence"], required=False)
        mean_col = find_col(df, ["mean_prob"], required=False)

        df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
        df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
        if score_col:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

        df = df.dropna(subset=[start_col, end_col]).copy()
        df["dur_s"] = df[end_col] - df[start_col]
        df = df[(df["dur_s"] >= args.min_dur_s) & (df["dur_s"] <= args.max_dur_s)]

        if score_col:
            df = df[df[score_col] >= args.min_score].copy()
            df = df.sort_values(score_col, ascending=False)
        else:
            df = df.sort_values(start_col)

        if args.topk_per_file > 0:
            df = df.head(args.topk_per_file)

        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            start_s = max(0.0, float(row[start_col]) - args.pad_s)
            end_s = float(row[end_col]) + args.pad_s
            dur_s = max(0.05, end_s - start_s)

            y, sr = librosa.load(
                str(wav_path),
                sr=args.sr,
                mono=True,
                offset=start_s,
                duration=dur_s,
            )

            if y is None or len(y) == 0:
                print(f"[WARN] empty clip: {wav_path} {start_s:.2f}-{end_s:.2f}")
                continue

            score = float(row[score_col]) if score_col else -1.0
            mean_prob = float(row[mean_col]) if mean_col else -1.0

            out_name = (
                f"{safe_name(stem)}"
                f"__rank{rank:03d}"
                f"__p{score:.3f}"
                f"__{start_s:.2f}-{end_s:.2f}.wav"
            )
            out_path = out_dir / out_name
            sf.write(str(out_path), y, args.sr)

            records.append({
                "clip_path": str(out_path.resolve()),
                "source_wav": str(wav_path.resolve()),
                "event_csv": str(event_csv.resolve()),
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "dur_s": round(dur_s, 4),
                "max_prob": round(score, 6),
                "mean_prob": round(mean_prob, 6),
                "rank_in_file": rank,
            })

        print(f"[OK] {stem}: exported {len(df)} clips")

    manifest = out_dir / "round4_candidates.csv"
    pd.DataFrame(records).to_csv(manifest, index=False, encoding="utf-8-sig")
    print(f"[DONE] total clips={len(records)}")
    print(f"[DONE] manifest={manifest}")


if __name__ == "__main__":
    main()