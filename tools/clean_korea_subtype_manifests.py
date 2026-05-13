from pathlib import Path
import pandas as pd
import librosa

ROOT = Path(r"C:\py\pigsound\pig-sound-classification")
MANI_DIR = ROOT / "data" / "manifests_korea_subtype"
OUT_DIR = ROOT / "data" / "manifests_korea_subtype_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
SR = 32000

def clean_one(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)
    good_rows = []
    bad_rows = []

    for _, row in df.iterrows():
        p = str(row["filepath"]).strip()
        lab = str(row["label"]).strip()
        try:
            y, sr = librosa.load(p, sr=SR, mono=True)
            if y is None or len(y) == 0:
                raise RuntimeError("zero-length audio")
            good_rows.append({"filepath": p, "label": lab})
        except Exception as e:
            bad_rows.append({"filepath": p, "label": lab, "error": repr(e)})

    good_df = pd.DataFrame(good_rows)
    bad_df = pd.DataFrame(bad_rows)

    good_df.to_csv(OUT_DIR / f"{split_name}.csv", index=False, encoding="utf-8-sig")
    bad_df.to_csv(OUT_DIR / f"{split_name}_bad.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] {split_name}: good={len(good_df)} bad={len(bad_df)}")

for split in SPLITS:
    clean_one(MANI_DIR / f"{split}.csv", split)