from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

SOURCES = [
    (ROOT / "data" / "raw" / "soundwel", "soundwel"),
    (ROOT / "data" / "raw" / "wu_pig_speech", "wu_pig_speech"),
]

rows = []

for folder, source in SOURCES:
    if not folder.exists():
        print("[WARN] missing:", folder)
        continue

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]

    print("[INFO]", source, "files =", len(files))

    for p in files:
        rows.append({
            "filepath": str(p.resolve()),
            "label": "other",
            "source": source,
        })

out_dir = ROOT / "data" / "external_pig_other_candidates"
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "soundwel_wu_all_other.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

print("[OK] wrote ->", out_csv)
print("[INFO] total =", len(rows))