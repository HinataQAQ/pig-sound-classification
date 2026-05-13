import pandas as pd
import librosa

df = pd.read_csv(r"data\manifests_cough_vs_piggrunt\test_head20.csv")

for i, r in df.iterrows():
    p = r["filepath"]
    print(i, p)
    y, sr = librosa.load(p, sr=32000, mono=True, duration=1.0)
    print("  ok", len(y), sr)