import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe(s: str) -> str:
    return re.sub(r'[^0-9a-zA-Z._-]+', '_', str(s))


def greedy_topk(
    df: pd.DataFrame,
    score_col: str,
    center_col: str,
    topk: int,
    min_separation: float,
    min_score: float,
    max_per_event: int = 2,
):
    work = df.copy()
    work = work[np.isfinite(work[score_col].to_numpy())]
    work = work[work[score_col] >= float(min_score)]
    work = work.sort_values(score_col, ascending=False).reset_index(drop=True)

    keep = []
    chosen_centers = []
    event_counter = {}

    for _, row in work.iterrows():
        c = float(row[center_col])
        event_id = int(row["event_id"]) if "event_id" in row.index else -1

        if event_id >= 0:
            cnt = event_counter.get(event_id, 0)
            if cnt >= int(max_per_event):
                continue

        if all(abs(c - prev) >= float(min_separation) for prev in chosen_centers):
            keep.append(row)
            chosen_centers.append(c)

            if event_id >= 0:
                event_counter[event_id] = event_counter.get(event_id, 0) + 1

            if len(keep) >= int(topk):
                break

    if not keep:
        return pd.DataFrame(columns=list(df.columns))

    out = pd.DataFrame(keep).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True, help='long wav path')
    ap.add_argument('--frames', required=True, help='*_frames.csv from detect_long_cough.py')
    ap.add_argument('--out_dir', default='det_round1_clips', help='output folder')
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--min_prob', type=float, default=0.50)
    ap.add_argument('--clip_seconds', type=float, default=1.0, help='export fixed-length clips of this duration')
    ap.add_argument('--window_seconds', type=float, default=1.0, help='original detector window length')
    ap.add_argument('--min_separation', type=float, default=0.60, help='greedy de-dup based on clip center distance')
    ap.add_argument('--prefix', type=str, default='round1')
    ap.add_argument("--events", type=str, default=None)
    ap.add_argument("--max_per_event", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.frames)
    event_df = None
    if args.events is not None:
        event_df = pd.read_csv(args.events)

    def assign_event_id(center_s, event_df):
        if event_df is None or len(event_df) == 0:
            return -1
        hit = event_df[(event_df["start_s"] <= center_s) & (event_df["end_s"] >= center_s)]
        if len(hit) == 0:
            return -1
        return int(hit.index[0])

    df["center_s"] = df["start_s"] + args.window_seconds / 2.0
    df["event_id"] = df["center_s"].apply(lambda x: assign_event_id(x, event_df))
    start_col = _find_col(df, ['start_s', 'start_sec', 'start'])
    end_col = _find_col(df, ['end_s', 'end_sec', 'end'])
    prob_col = _find_col(df, ['score_smooth', 'score_final', 'prob_cough_smooth', 'prob_cough'])
    if start_col is None or prob_col is None:
        raise KeyError(f'frames csv must contain start_s and prob_cough/prob_cough_smooth, got {list(df.columns)}')

    if end_col is None:
        df['end_s'] = df[start_col].astype(float) + float(args.window_seconds)
        end_col = 'end_s'

    df['center_s'] = 0.5 * (df[start_col].astype(float) + df[end_col].astype(float))
    picked = greedy_topk(
        df=df,
        score_col=prob_col,
        center_col='center_s',
        topk=int(args.topk),
        min_separation=float(args.min_separation),
        min_score=float(args.min_prob),
        max_per_event=int(args.max_per_event),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []

    with sf.SoundFile(args.wav, 'r') as f:
        sr = int(f.samplerate)
        total_frames = int(len(f))

        for _, row in picked.iterrows():
            center = float(row['center_s'])
            score = float(row[prob_col])
            clip_start = max(0.0, center - float(args.clip_seconds) / 2.0)
            clip_end = clip_start + float(args.clip_seconds)

            s_idx = int(round(clip_start * sr))
            e_idx = int(round(clip_end * sr))
            if e_idx > total_frames:
                e_idx = total_frames
                s_idx = max(0, e_idx - int(round(float(args.clip_seconds) * sr)))
            if e_idx <= s_idx:
                continue

            f.seek(s_idx)
            clip = f.read(e_idx - s_idx, dtype='float32', always_2d=False)
            if isinstance(clip, np.ndarray) and clip.ndim == 2:
                clip = clip.mean(axis=1).astype(np.float32)

            fname = (
                f"{_safe(args.prefix)}_rank{int(row['rank']):03d}"
                f"_p{score:.3f}_src{float(row[start_col]):.2f}-{float(row[end_col]):.2f}"
                f"_clip{clip_start:.2f}-{clip_end:.2f}.wav"
            )
            out_wav = out_dir / fname
            sf.write(out_wav, clip, sr)
            meta_rows.append(
                {
                    'rank': int(row['rank']),
                    'source_wav': os.path.abspath(args.wav),
                    'frame_csv': os.path.abspath(args.frames),
                    'source_start_s': float(row[start_col]),
                    'source_end_s': float(row[end_col]),
                    'center_s': center,
                    'clip_start_s': clip_start,
                    'clip_end_s': clip_end,
                    'prob_cough': score,
                    'out_wav': str(out_wav.resolve()),
                }
            )

    out_csv = out_dir / f'{_safe(args.prefix)}_topk_metadata.csv'
    pd.DataFrame(meta_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print('[OK] selected =', len(meta_rows))
    print('[OK] out_dir  =', str(out_dir.resolve()))
    print('[OK] meta_csv =', str(out_csv.resolve()))


if __name__ == '__main__':
    main()
