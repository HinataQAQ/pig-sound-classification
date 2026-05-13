from pathlib import Path
import torch

CKPT_DIR = Path(r"C:\py\pigsound\pig-sound-classification\checkpoints")

for p in sorted(CKPT_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
    try:
        ckpt = torch.load(str(p), map_location="cpu")
        cfg = ckpt.get("config", {})
        data = cfg.get("data", {})
        exp = cfg.get("experiment_name", "")
        train_manifest = data.get("train_manifest", "")
        val_manifest = data.get("val_manifest", "")
        test_manifest = data.get("test_manifest", "")

        print("=" * 90)
        print("ckpt:", p.name)
        print("experiment:", exp)
        print("train_manifest:", train_manifest)
        print("val_manifest:  ", val_manifest)
        print("test_manifest: ", test_manifest)
    except Exception as e:
        print("=" * 90)
        print("ckpt:", p.name)
        print("ERROR:", repr(e))