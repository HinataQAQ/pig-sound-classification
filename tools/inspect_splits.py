import pandas as pd

for s in ["train", "val", "test"]:
    df = pd.read_csv(f"data/manifests/{s}.csv")
    print("\n===", s, "===")
    print("N =", len(df))
    print(df["label"].value_counts())
