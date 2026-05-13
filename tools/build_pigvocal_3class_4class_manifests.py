from pathlib import Path
import argparse
import random
import pandas as pd
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def collect_audio(folder: Path):
    if not folder.exists():
        print(f"[WARN] missing folder: {folder}")
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def is_valid_audio(path: Path) -> bool:
    try:
        info = sf.info(str(path))
        return info.frames > 0
    except Exception:
        return False


def sample_valid(files, n, rng, tag):
    files = list(files)
    rng.shuffle(files)

    kept = []
    bad = []

    for p in files:
        if len(kept) >= n:
            break
        if is_valid_audio(p):
            kept.append(p)
        else:
            bad.append(p)

    print(f"[INFO] {tag}: requested={n}, kept={len(kept)}, bad={len(bad)}")
    return kept, bad


def add_rows(rows, files, label, source):
    for p in files:
        rows.append({
            "filepath": str(p.resolve()),
            "label": label,
            "source": source,
        })


def split_stratified(rows, rng, train_ratio=0.8, val_ratio=0.1):
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, val, test = [], [], []

    for lab, items in by_label.items():
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_split(out_dir: Path, name: str, rows):
    df = pd.DataFrame(rows)
    df[["filepath", "label"]].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out_dir / f"{name}_full.csv", index=False, encoding="utf-8-sig")

    print(f"\n[OK] {name}: {len(df)}")
    print(df["label"].value_counts())
    print(df["source"].value_counts())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["3class", "4class"], required=True)

    ap.add_argument("--korea_dry_dir", required=True)
    ap.add_argument("--korea_abdominal_dir", required=True)
    ap.add_argument("--sow_labeled_dir", required=True)
    ap.add_argument("--reviewed_grunt_dir", default="")

    ap.add_argument("--n_per_class", type=int, default=350)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    bad_rows = []

    korea_dry = collect_audio(Path(args.korea_dry_dir))
    korea_abd = collect_audio(Path(args.korea_abdominal_dir))

    cough_all = korea_dry + korea_abd
    cough_keep, cough_bad = sample_valid(cough_all, args.n_per_class, rng, "cough")
    add_rows(rows, cough_keep, "cough", "korea_cough")
    for p in cough_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "korea_cough"})

    sow = Path(args.sow_labeled_dir)

    if args.mode == "3class":
        calm_files = collect_audio(sow / "calm_grunt")
        reviewed_grunt = collect_audio(Path(args.reviewed_grunt_dir)) if args.reviewed_grunt_dir else []
        grunt_all = calm_files + reviewed_grunt

        stress_all = collect_audio(sow / "frightened_stress") + collect_audio(sow / "anxious_stress")

        grunt_keep, grunt_bad = sample_valid(grunt_all, args.n_per_class, rng, "grunt_oink")
        stress_keep, stress_bad = sample_valid(stress_all, args.n_per_class, rng, "stress_vocal")

        add_rows(rows, grunt_keep, "grunt_oink", "sow_calm_plus_reviewed_grunt")
        add_rows(rows, stress_keep, "stress_vocal", "sow_frightened_anxious")

        for p in grunt_bad:
            bad_rows.append({"filepath": str(p.resolve()), "source": "grunt_oink"})
        for p in stress_bad:
            bad_rows.append({"filepath": str(p.resolve()), "source": "stress_vocal"})

    else:
        calm_keep, calm_bad = sample_valid(collect_audio(sow / "calm_grunt"), args.n_per_class, rng, "calm_grunt")
        feeding_keep, feeding_bad = sample_valid(collect_audio(sow / "feeding"), args.n_per_class, rng, "feeding")
        stress_all = collect_audio(sow / "frightened_stress") + collect_audio(sow / "anxious_stress")
        stress_keep, stress_bad = sample_valid(stress_all, args.n_per_class, rng, "stress_vocal")

        add_rows(rows, calm_keep, "calm_grunt", "sow_calm")
        add_rows(rows, feeding_keep, "feeding", "sow_feeding")
        add_rows(rows, stress_keep, "stress_vocal", "sow_frightened_anxious")

        for p in calm_bad:
            bad_rows.append({"filepath": str(p.resolve()), "source": "calm_grunt"})
        for p in feeding_bad:
            bad_rows.append({"filepath": str(p.resolve()), "source": "feeding"})
        for p in stress_bad:
            bad_rows.append({"filepath": str(p.resolve()), "source": "stress_vocal"})

    # remove exact duplicate path
    df = pd.DataFrame(rows)
    df["_norm"] = df["filepath"].map(lambda x: str(Path(x).resolve()).lower())
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"]).copy()
    rows = df.to_dict("records")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = split_stratified(rows, rng)

    write_split(out_dir, "train", train)
    write_split(out_dir, "val", val)
    write_split(out_dir, "test", test)

    pd.DataFrame(bad_rows).to_csv(out_dir / "bad_audio.csv", index=False, encoding="utf-8-sig")
    print("\n[DONE] wrote ->", out_dir)


if __name__ == "__main__":
    main()