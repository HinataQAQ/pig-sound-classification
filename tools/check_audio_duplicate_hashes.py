from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parents[1]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

TARGET_DIRS = [
    ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "calm_grunt",
    ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "feeding",
    ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "frightened_stress",
    ROOT / "data" / "raw" / "sow_call_dataset_labeled" / "anxious_stress",
    ROOT / "data" / "external" / "korea_raw" / "dry_cough",
    ROOT / "data" / "external" / "korea_raw" / "abdominal_cough",
]

def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    files = []
    for d in TARGET_DIRS:
        if not d.exists():
            print(f"[MISSING] {d}")
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)

    print(f"[INFO] total audio files = {len(files)}")

    by_hash = defaultdict(list)

    for i, p in enumerate(files, 1):
        if i % 500 == 0:
            print(f"[INFO] hashing {i}/{len(files)}")
        by_hash[md5_file(p)].append(p)

    dup_groups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}
    dup_files = sum(len(ps) for ps in dup_groups.values())

    print(f"\n[RESULT]")
    print(f"unique hashes      = {len(by_hash)}")
    print(f"duplicate groups   = {len(dup_groups)}")
    print(f"duplicate files    = {dup_files}")

    if dup_groups:
        print("\n[EXAMPLES]")
        for j, (h, ps) in enumerate(dup_groups.items()):
            if j >= 20:
                break
            print(f"\nMD5 = {h}")
            for p in ps:
                print("  ", p.relative_to(ROOT))

if __name__ == "__main__":
    main()