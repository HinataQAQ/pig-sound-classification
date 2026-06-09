from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import random
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

CLASS_SOURCES = {
    "cough": [
        ("dry_cough", ROOT / "data" / "external" / "korea_raw" / "dry_cough"),
        ("abdominal_cough", ROOT / "data" / "external" / "korea_raw" / "abdominal_cough"),
    ],
    "calm_grunt": [
        ("calm_grunt", ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "calm_grunt"),
    ],
    "feeding": [
        ("feeding", ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "feeding"),
    ],
    "stress_vocal": [
        ("frightened_stress", ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "frightened_stress"),
        ("anxious_stress", ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "anxious_stress"),
    ],
}


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_id_from_path(path: Path) -> str:
    s = path.stem.lower()
    s = re.sub(r"__dup$", "", s)

    patterns = [
        r"[_\-]chunk[_\-]?\d+$",
        r"[_\-]seg[_\-]?\d+$",
        r"[_\-]slice[_\-]?\d+$",
        r"[_\-]part[_\-]?\d+$",
    ]

    for pat in patterns:
        s = re.sub(pat, "", s)

    return f"{path.parent.name}::{s}"


def canonical_key(path: Path):
    stem = path.stem.lower()
    is_dup = "__dup" in stem
    return (is_dup, len(str(path)), str(path))


def collect_unique_audio():
    rows = []

    for label, sources in CLASS_SOURCES.items():
        for subtype, directory in sources:
            if not directory.exists():
                print(f"[MISSING] {directory}")
                continue

            files = [
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            ]

            print(f"[COLLECT] {label:12s} {subtype:20s} raw_files={len(files)}")

            by_hash = defaultdict(list)

            for i, p in enumerate(files, 1):
                if i % 1000 == 0:
                    print(f"  hashing {label}/{subtype}: {i}/{len(files)}")
                by_hash[md5_file(p)].append(p)

            for h, ps in by_hash.items():
                p = sorted(ps, key=canonical_key)[0]
                rows.append({
                    "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "label": label,
                    "subtype": subtype,
                    "source_id": source_id_from_path(p),
                    "md5": h,
                })

            print(
                f"[UNIQUE]  {label:12s} {subtype:20s} "
                f"unique={len(by_hash)} duplicates_removed={len(files)-len(by_hash)}"
            )

    return rows


def read_manifest(path: Path):
    df = pd.read_csv(path)
    rows = df.to_dict("records")

    out = []

    for r in rows:
        p = ROOT / str(r["path"]).replace("\\", "/")
        if "md5" in r and isinstance(r["md5"], str) and len(r["md5"]) > 0:
            h = r["md5"]
        else:
            h = md5_file(p)

        source_id = r.get("source_id")
        if not isinstance(source_id, str) or len(source_id) == 0:
            source_id = source_id_from_path(p)

        out.append({
            "path": str(Path(str(r["path"])).as_posix()),
            "label": str(r["label"]),
            "subtype": str(r.get("subtype", "")),
            "source_id": source_id,
            "md5": h,
        })

    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["path", "label", "subtype", "source_id", "md5"]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def count_labels(rows):
    d = defaultdict(int)
    for r in rows:
        d[r["label"]] += 1
    return dict(d)


def count_subtypes(rows):
    d = defaultdict(int)
    for r in rows:
        d[f"{r['label']}::{r.get('subtype', '')}"] += 1
    return dict(d)


def select_train_rows(candidates, mode, rng, cap_multiplier):
    by_label = defaultdict(list)
    for r in candidates:
        by_label[r["label"]].append(r)

    for label in by_label:
        rng.shuffle(by_label[label])

    if mode == "all":
        selected = []
        for label in ["cough", "calm_grunt", "feeding", "stress_vocal"]:
            selected.extend(by_label[label])
        rng.shuffle(selected)
        return selected

    if mode == "cap3x":
        feeding_n = len(by_label["feeding"])
        cap = int(feeding_n * cap_multiplier)

        selected = []
        for label in ["cough", "calm_grunt", "feeding", "stress_vocal"]:
            rows = by_label[label]
            if label == "feeding":
                take = len(rows)
            else:
                take = min(len(rows), cap)
            selected.extend(rows[:take])

        rng.shuffle(selected)
        return selected

    raise ValueError(f"Unknown mode={mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cv5", default="data/manifests_pigvocal_4class_dedup_cv5")
    ap.add_argument("--out_root", default="data/manifests_pigvocal_4class_expanded_train_cv5_cap3x")
    ap.add_argument("--mode", choices=["cap3x", "all"], default="cap3x")
    ap.add_argument("--cap_multiplier", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    base_cv5 = ROOT / args.base_cv5
    out_root = ROOT / args.out_root

    all_unique = collect_unique_audio()

    print("\n[ALL UNIQUE LABEL COUNTS]")
    print(count_labels(all_unique))
    print("[ALL UNIQUE SUBTYPE COUNTS]")
    print(count_subtypes(all_unique))

    for fold in range(5):
        print(f"\n===== FOLD {fold} =====")

        base_fold = base_cv5 / f"fold{fold}"

        val_rows = read_manifest(base_fold / "val.csv")
        test_rows = read_manifest(base_fold / "test.csv")

        heldout_md5 = {r["md5"] for r in val_rows + test_rows}
        heldout_source = {r["source_id"] for r in val_rows + test_rows}

        candidates = []
        for r in all_unique:
            if r["md5"] in heldout_md5:
                continue
            if r["source_id"] in heldout_source:
                continue
            candidates.append(r)

        train_rows = select_train_rows(
            candidates=candidates,
            mode=args.mode,
            rng=rng,
            cap_multiplier=args.cap_multiplier,
        )

        out_fold = out_root / f"fold{fold}"
        write_csv(out_fold / "train.csv", train_rows)
        write_csv(out_fold / "val.csv", val_rows)
        write_csv(out_fold / "test.csv", test_rows)

        print("train:", len(train_rows), count_labels(train_rows))
        print("val:  ", len(val_rows), count_labels(val_rows))
        print("test: ", len(test_rows), count_labels(test_rows))
        print("train subtypes:", count_subtypes(train_rows))

    print(f"\n[OK] wrote -> {out_root}")


if __name__ == "__main__":
    main()