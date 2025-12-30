# tools/make_overfit_manifest.py
import pandas as pd
from pathlib import Path

src = Path("data/manifests_cough_binary/train.csv")
df = pd.read_csv(src)
df["label"] = df["label"].astype(str).str.strip().str.lower()

# 每类抽 32 条 => 总共 64 条
sub = df.groupby("label", group_keys=False).apply(lambda x: x.sample(32, random_state=0))

out = Path("data/manifests_cough_binary/overfit64.csv")
sub.to_csv(out, index=False)

print(sub["label"].value_counts())
print("Wrote:", out)
