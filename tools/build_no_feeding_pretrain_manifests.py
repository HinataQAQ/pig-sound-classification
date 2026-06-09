from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import random
import re

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}


CLASS_SOURCES = {
    "dry_cough": [
        ROOT / "data" / "external" / "korea_raw" / "dry_cough",
    ],
    "abdominal_cough": [
        ROOT / "data" / "external" / "korea_raw" / "abdominal_cough",
    ],
    "calm_grunt": [
        ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "calm_grunt",
    ],
    "frightened_stress": [
        ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "frightened_stress",
    ],
    "anxious_stress": [
        ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "anxious_stress",
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

    # 去掉显式重复后缀
    s = re.sub(r"__dup$", "", s)

    # 去掉常见切片编号后缀
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
    如果同一个 md5 有原始文件和 __dup 文件，优先保留非 __dup 文件。
    """
    stem = path.stem.lower()
    is_dup = "__dup" in stem
    return (is_dup, len(str(path)), str(path))


def collect_unique(label: str, directories):
    by_hash = defaultdict(list)

    raw_files = []

    for d in directories:
        if not d.exists():
            print(f"[MISSING] {d}")
            continue

        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                raw_files.append(p)

    print(f"[COLLECT] {label:18s} raw_files={len(raw_files)}")

    for i, p in enumerate(raw_files, 1):
        if i % 1000 == 0:
            print(f"  hashing {label}: {i}/{len(raw_files)}")
        h = md5_file(p)
        by_hash[h].append(p)

    rows = []

    for h, ps in by_hash.items():
        p = sorted(ps, key=canonical_key)[0]
        rows.append({
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "label": label,
            "subtype": label,
            "source_id": source_id_from_path(p),
            "md5": h,
        })

    rows = sorted(rows, key=lambda r: r["path"])

    print(
        f"[UNIQUE]  {label:18s} unique={len(rows)} "
        f"duplicates_removed={len(raw_files) - len(rows)}"
    )

    return rows


def group_by_source(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_id"]].append(r)
    return list(groups.values())


def split_source_disjoint(rows, rng, val_ratio=0.1, test_ratio=0.1):
    """
    source-disjoint split:
    同一个 source_id 的样本不会跨 train / val / test。
    """
    groups = group_by_source(rows)
    rng.shuffle(groups)

    n_total = len(rows)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))

    train, val, test = [], [], []

    for g in groups:
        if len(test) < n_test:
            test.extend(g)
        elif len(val) < n_val:
            val.extend(g)
        else:
            train.extend(g)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/manifests_pigvocal_no_feeding_pretrain_cap1000",
    )

    ap.add_argument(
        "--mode",
        type=str,
        default="cap",
        choices=["cap", "balanced"],
        help=(
            "cap: 每类最多 max_per_class 条；"
            "balanced: 所有类取相同数量，数量等于最小类。"
        ),
    )

    ap.add_argument("--max_per_class", type=int, default=1000)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)

    args = ap.parse_args()

    rng = random.Random(args.seed)

    all_by_label = {}

    for label, dirs in CLASS_SOURCES.items():
        rows = collect_unique(label, dirs)
        rng.shuffle(rows)
        all_by_label[label] = rows

    print("\n[RAW UNIQUE COUNTS]")
    for label, rows in all_by_label.items():
        print(f"{label:18s} {len(rows)}")

    if args.mode == "balanced":
        n_take = min(len(rows) for rows in all_by_label.values())
        print(f"\n[MODE] balanced, n_take per class = {n_take}")
    else:
        n_take = args.max_per_class
        print(f"\n[MODE] cap, max_per_class = {n_take}")

    train_all, val_all, test_all = [], [], []

    for label, rows in all_by_label.items():
        take = min(len(rows), n_take)
        selected = rows[:take]

        train, val, test = split_source_disjoint(
            selected,
            rng=rng,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)

        print(
            f"[SPLIT] {label:18s} "
            f"selected={len(selected):5d} "
            f"train={len(train):5d} val={len(val):4d} test={len(test):4d}"
        )

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    out_dir = ROOT / args.out_dir

    write_csv(out_dir / "train.csv", train_all)
    write_csv(out_dir / "val.csv", val_all)
    write_csv(out_dir / "test.csv", test_all)

    print("\n[WROTE]")
    print(out_dir / "train.csv")
    print(out_dir / "val.csv")
    print(out_dir / "test.csv")

    print("\n[LABEL COUNTS]")
    print("train:", count_labels(train_all))
    print("val:  ", count_labels(val_all))
    print("test: ", count_labels(test_all))


if __name__ == "__main__":
    main()