from pathlib import Path
import pandas as pd
import librosa

MANIFEST = Path(r"C:\py\pigsound\pig-sound-classification\data\manifests_korea\korea_holdout_pos_round1.csv")
OUT_GOOD = Path(r"C:\py\pigsound\pig-sound-classification\data\manifests_korea\korea_holdout_pos_round1_clean.csv")
OUT_BAD = Path(r"C:\py\pigsound\pig-sound-classification\data\manifests_korea\korea_holdout_pos_round1_bad.csv")

df = pd.read_csv(MANIFEST)
path_col = "filepath" if "filepath" in df.columns else "path"

good_rows = []
bad_rows = []

for _, row in df.iterrows():
    p = str(row[path_col]).strip().strip('"')
    try:
        if not Path(p).exists():
            raise FileNotFoundError(p)

        y, sr = librosa.load(p, sr=32000, mono=True)

        if y is None or len(y) == 0:
            raise RuntimeError("zero-length audio")

        good_rows.append(row.to_dict())
    except Exception as e:
        r = row.to_dict()
        r["error"] = repr(e)
        bad_rows.append(r)
        print("[BAD]", p, repr(e))

pd.DataFrame(good_rows).to_csv(OUT_GOOD, index=False, encoding="utf-8-sig")
pd.DataFrame(bad_rows).to_csv(OUT_BAD, index=False, encoding="utf-8-sig")

print("[OK] good =", len(good_rows), "->", OUT_GOOD)
print("[OK] bad  =", len(bad_rows), "->", OUT_BAD)