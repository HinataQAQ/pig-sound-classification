from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# 1) 原始二分类主数据
BASE_DIRS = [
    (ROOT / "data" / "processed" / "cough_silence_binary" / "cough", "cough", "base"),
    (ROOT / "data" / "processed" / "cough_silence_binary" / "other", "other", "base"),
]

# 2) 人工复核后的 clean 数据
# 会递归扫描 det_round1_clips / det_round2_clips 下所有 cough_clean / other_clean
REVIEW_ROOTS = [
    ROOT / "det_round1_clips",
    ROOT / "det_round2_clips",
]

def collect_audio_files(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

rows = []

# 原始主数据
for folder, label, source in BASE_DIRS:
    for p in collect_audio_files(folder):
        rows.append({
            "filepath": str(p.resolve()),
            "label": label,
            "source": source,
        })

# review clean 数据
for rr in REVIEW_ROOTS:
    if not rr.exists():
        continue

    for p in rr.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue

        parts_lower = [x.lower() for x in p.parts]

        if "cough_clean" in parts_lower:
            label = "cough"
            source = "review_clean"
        elif "other_clean" in parts_lower:
            label = "other"
            source = "review_clean"
        else:
            continue

        rows.append({
            "filepath": str(p.resolve()),
            "label": label,
            "source": source,
        })

df = pd.DataFrame(rows)

# 去重：同一路径只保留一次
df = df.drop_duplicates(subset=["filepath"]).copy()

# 只导出当前训练真正需要的两列
out_dir = ROOT / "data" / "manifests_rebuilt"
out_dir.mkdir(parents=True, exist_ok=True)

out_full = out_dir / "train_rebuilt_full.csv"
out_train = out_dir / "train_rebuilt.csv"

df.to_csv(out_full, index=False, encoding="utf-8-sig")
df[["filepath", "label"]].to_csv(out_train, index=False, encoding="utf-8-sig")

print("[OK] wrote ->", out_full)
print("[OK] wrote ->", out_train)
print(df["label"].value_counts())
print("total =", len(df))