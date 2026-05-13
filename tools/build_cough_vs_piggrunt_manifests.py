from pathlib import Path
import argparse
import random
import pandas as pd
import librosa


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def collect_audio(folder: Path):
    if not folder.exists():
        print(f"[WARN] missing folder: {folder}")
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ])


def is_valid_audio(path: Path, sr=32000) -> bool:
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=0.2)
        return y is not None and len(y) > 0
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

    print(f"[INFO] {tag}: requested={n}, kept={len(kept)}, bad_checked={len(bad)}")
    return kept, bad


def split_rows(rows, rng, train_ratio=0.8, val_ratio=0.1):
    cough = [r for r in rows if r["label"] == "cough"]
    other = [r for r in rows if r["label"] == "other"]

    rng.shuffle(cough)
    rng.shuffle(other)

    def split_one(lst):
        n = len(lst)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]
        return train, val, test

    c_tr, c_va, c_te = split_one(cough)
    o_tr, o_va, o_te = split_one(other)

    train = c_tr + o_tr
    val = c_va + o_va
    test = c_te + o_te

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--korea_dry_dir", required=True)
    ap.add_argument("--korea_abdominal_dir", required=True)
    ap.add_argument("--soundwel_dir", required=True)
    ap.add_argument("--wu_dir", required=True)
    ap.add_argument("--reviewed_grunt_dir", default="")

    ap.add_argument("--n_korea_dry", type=int, default=1500)
    ap.add_argument("--n_korea_abdominal", type=int, default=1500)
    ap.add_argument("--n_soundwel", type=int, default=1500)
    ap.add_argument("--n_wu", type=int, default=1000)
    ap.add_argument("--n_reviewed_grunt", type=int, default=300)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    dry_files = collect_audio(Path(args.korea_dry_dir))
    abd_files = collect_audio(Path(args.korea_abdominal_dir))
    soundwel_files = collect_audio(Path(args.soundwel_dir))
    wu_files = collect_audio(Path(args.wu_dir))
    reviewed_files = collect_audio(Path(args.reviewed_grunt_dir)) if args.reviewed_grunt_dir else []

    print("[INFO] total dry =", len(dry_files))
    print("[INFO] total abdominal =", len(abd_files))
    print("[INFO] total soundwel =", len(soundwel_files))
    print("[INFO] total wu =", len(wu_files))
    print("[INFO] total reviewed_grunt =", len(reviewed_files))

    dry_keep, dry_bad = sample_valid(dry_files, args.n_korea_dry, rng, "korea_dry")
    abd_keep, abd_bad = sample_valid(abd_files, args.n_korea_abdominal, rng, "korea_abdominal")
    soundwel_keep, soundwel_bad = sample_valid(soundwel_files, args.n_soundwel, rng, "soundwel_other")
    wu_keep, wu_bad = sample_valid(wu_files, args.n_wu, rng, "wu_other")
    reviewed_keep, reviewed_bad = sample_valid(reviewed_files, args.n_reviewed_grunt, rng, "reviewed_grunt")

    rows = []

    for p in dry_keep:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "cough",
            "source": "korea_dry_cough",
        })

    for p in abd_keep:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "cough",
            "source": "korea_abdominal_cough",
        })

    for p in soundwel_keep:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "other",
            "source": "soundwel_pig_vocal_other",
        })

    for p in wu_keep:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "other",
            "source": "wu_pig_speech_other",
        })

    for p in reviewed_keep:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "other",
            "source": "reviewed_pig_grunt_oink",
        })

    # 去掉重复路径
    df = pd.DataFrame(rows)
    df["_norm"] = df["filepath"].map(lambda x: str(Path(x).resolve()).lower())
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"]).copy()

    rows = df.to_dict("records")
    train, val, test = split_rows(rows, rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part_df = pd.DataFrame(part)
        part_df[["filepath", "label"]].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        part_df.to_csv(out_dir / f"{name}_full.csv", index=False, encoding="utf-8-sig")

        print(f"[OK] {name}: {len(part_df)}")
        print(part_df["label"].value_counts())
        print(part_df["source"].value_counts())

    bad_rows = []
    for p in dry_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "korea_dry_cough"})
    for p in abd_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "korea_abdominal_cough"})
    for p in soundwel_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "soundwel_pig_vocal_other"})
    for p in wu_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "wu_pig_speech_other"})
    for p in reviewed_bad:
        bad_rows.append({"filepath": str(p.resolve()), "source": "reviewed_pig_grunt_oink"})

    pd.DataFrame(bad_rows).to_csv(out_dir / "bad_audio.csv", index=False, encoding="utf-8-sig")

    print("[DONE] wrote ->", out_dir)


if __name__ == "__main__":
    main()