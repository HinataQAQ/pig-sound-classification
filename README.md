# pig-sound-lab
一个**可直接运行**的生猪声音分类项目骨架（Python / PyTorch，适配 PyCharm）。
默认实现 **MFCC + GFCC 融合特征 + BiLSTM 分类器**，并提供数据清洗与增广的最小实现。

## 快速开始
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 准备 data/manifests/train.csv 和 val.csv 两列: filepath,label
python -m src.train --config config/config.yaml
python -m src.infer --model checkpoints/model.pt --wav your_audio.wav
```

> 项目默认**不依赖任何专用硬件**，CPU 可运行；若有 GPU，PyTorch 会自动利用。
