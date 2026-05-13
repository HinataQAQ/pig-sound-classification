import torch

ckpt_path = r"checkpoints\cough_silence_crnn_transfer_from_korea_clean_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
cfg = ckpt.get("config", {})

print("experiment_name =", cfg.get("experiment_name"))
print("train_manifest =", cfg.get("data", {}).get("train_manifest"))
print("val_manifest   =", cfg.get("data", {}).get("val_manifest"))
print("test_manifest  =", cfg.get("data", {}).get("test_manifest"))