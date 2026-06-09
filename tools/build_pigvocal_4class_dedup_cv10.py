from pathlib import Path
from collections import defaultdict
import hashlib
import random
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

SEED = 3407
N_FOLDS = 10

# Total unique samples per class.
# Feeding is the limiting class.
N_PER_CLASS = 210

# Fold-level split size per class:
# 210 = 42 test + 30 val + 138 train
TEST_N = 21
VAL_N = 21

OUT_ROOT = ROOT / "data" / "manifests_pigvocal_4class_dedup_cv10"

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

    # Remove explicit duplicate suffix.
    s = re.sub(r"__dup$", "", s)

    # Remove common segment/chunk suffixes.
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
    """
    Prefer non-__dup files, then shorter path.
    """
    stem = path.stem.lower()
    is_dup = "__dup" in stem
    return (is_dup, len(str(path)), str(path))


def collect_unique(label: str, subtype: str, directory: Path):
    if not directory.exists():
        raise FileNotFoundError(directory)

    by_hash = defaultdict(list)

    files = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]

    print(f"[COLLECT] {label:12s} {subtype:20s} raw_files={len(files)}")

    for i, p in enumerate(files, 1):
        if i % 500 == 0:
            print(f"  hashing {i}/{len(files)}")
        h = md5_file(p)
        by_hash[h].append(p)

    rows = []
    for h, ps in by_hash.items():
        p = sorted(ps, key=canonical_key)[0]
        rows.append({
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "label": label,
            "subtype": subtype,
            "source_id": source_id_from_path(p),
            "md5": h,
        })

    rows = sorted(rows, key=lambda r: r["path"])

    print(
        f"[UNIQUE]  {label:12s} {subtype:20s} "
        f"unique={len(rows)} duplicates_removed={len(files)-len(rows)}"
    )
    return rows


def select_balanced_pool(rng: random.Random):
    pools = {}

    for label, sources in CLASS_SOURCES.items():
        print(f"\n[CLASS] {label}")

        if label in {"cough", "stress_vocal"}:
            # Balance subtypes: 105 + 105 = 210
            each_n = N_PER_CLASS // 2
            selected = []

            for subtype, directory in sources:
                rows = collect_unique(label, subtype, directory)
                if len(rows) < each_n:
                    raise RuntimeError(
                        f"Not enough unique samples for {label}/{subtype}: "
                        f"{len(rows)} < {each_n}"
                    )
                rng.shuffle(rows)
                selected.extend(rows[:each_n])
                print(f"[SELECT] {label:12s} {subtype:20s} n={each_n}")

            rng.shuffle(selected)
            pools[label] = selected

        else:
            subtype, directory = sources[0]
            rows = collect_unique(label, subtype, directory)
            if len(rows) < N_PER_CLASS:
                raise RuntimeError(
                    f"Not enough unique samples for {label}: "
                    f"{len(rows)} < {N_PER_CLASS}"
                )
            rng.shuffle(rows)
            pools[label] = rows[:N_PER_CLASS]
            print(f"[SELECT] {label:12s} {subtype:20s} n={N_PER_CLASS}")

    return pools


def make_folds_for_class(rows, rng: random.Random):
    """
    Produces 5 disjoint test folds.
    Each fold gets TEST_N rows.
    """
    rows = list(rows)
    rng.shuffle(rows)

    folds = []
    for i in range(N_FOLDS):
        start = i * TEST_N
        end = start + TEST_N
        folds.append(rows[start:end])

    used = sum(len(f) for f in folds)
    if used != N_PER_CLASS:
        raise RuntimeError(f"Fold split mismatch: used={used}, expected={N_PER_CLASS}")

    return folds


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
        d[f"{r['label']}::{r['subtype']}"] += 1
    return dict(d)


def main():
    rng = random.Random(SEED)

    pools = select_balanced_pool(rng)

    class_folds = {}
    for label, rows in pools.items():
        class_folds[label] = make_folds_for_class(rows, rng)

    for fold_idx in range(N_FOLDS):
        train_rows = []
        val_rows = []
        test_rows = []

        for label in ["cough", "calm_grunt", "feeding", "stress_vocal"]:
            test = class_folds[label][fold_idx]
            rest = []
            for j in range(N_FOLDS):
                if j != fold_idx:
                    rest.extend(class_folds[label][j])

            rng.shuffle(rest)

            val = rest[:VAL_N]
            train = rest[VAL_N:]

            test_rows.extend(test)
            val_rows.extend(val)
            train_rows.extend(train)

        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        rng.shuffle(test_rows)

        out_dir = OUT_ROOT / f"fold{fold_idx}"
        write_csv(out_dir / "train.csv", train_rows)
        write_csv(out_dir / "val.csv", val_rows)
        write_csv(out_dir / "test.csv", test_rows)

        print(f"\n[FOLD {fold_idx}]")
        print("train:", len(train_rows), count_labels(train_rows))
        print("val:  ", len(val_rows), count_labels(val_rows))
        print("test: ", len(test_rows), count_labels(test_rows))
        print("subtypes test:", count_subtypes(test_rows))

    print(f"\n[OK] wrote folds -> {OUT_ROOT}")


if __name__ == "__main__":
    main()