# PyCharm 运行配置（Run/Debug Configurations）一键抄作业

> 将本文件夹里的 `tools/` 和 `src/eval.py` 复制到你的仓库根目录（与 `src/`, `config/`, `data/` 同级）。

## 需要的配置（每个都在 “Run → Edit Configurations… → + → Python” 新建）

### 01) 预处理：统一采样率并切成 2 秒片段
- **Script path**: `$ProjectFileDir$\tools\preprocess.py`
- **Parameters**: `--in_dir data\raw\soundwel --out_dir data\processed\soundwel --sr 16000 --clip_sec 2.0 --denoise`
- **Working directory**: `$ProjectFileDir$`
- **Python interpreter**: 选择你创建的 `pigsound` 环境

> 输入/输出目录按你实际修改；`--denoise` 可去掉。

### 02) 制作清单（train/val/test.csv）
- **Script path**: `$ProjectFileDir$\tools\make_manifests.py`
- **Parameters**: `--audio_root data\processed\soundwel --out_csv data\manifests --val_ratio 0.2 --test_ratio 0.2`
- **Working directory**: `$ProjectFileDir$`

> 终端会打印发现的 **标签列表**，把它们复制进 `config/config.yaml` 的 `data.labels`。

### 03) 训练（BiLSTM + MFCC/GFCC 融合）
- **Script path**: `$ProjectFileDir$\src\train.py`
- **Parameters**: `--config config\config.yaml`
- **Working directory**: `$ProjectFileDir$`

> 训练结束会把最佳权重保存到 `checkpoints\model.pt`。

### 04) 单文件推理
- **Script path**: `$ProjectFileDir$\src\infer.py`
- **Parameters**: `--model checkpoints\model.pt --wav 你的测试音频.wav`
- **Working directory**: `$ProjectFileDir$`

### 05) 在测试集上评估
- **Script path**: `$ProjectFileDir$\src\eval.py`
- **Parameters**: `--ckpt checkpoints\model.pt --manifest data\manifests\test.csv --config config\config.yaml`
- **Working directory**: `$ProjectFileDir$`

---

## 数据文件摆放规范（例）
```
data/
  raw/
    soundwel/
      cough/xxx.wav
      scream/yyy.wav
      grunt/zzz.wav
      ...
  processed/
    soundwel/        # 由 preprocess.py 生成
      cough/*.wav
      ...
  manifests/
    train.csv
    val.csv
    test.csv
```

## 小贴士
- Windows 下路径分隔符用 `\`，macOS/Linux 用 `/`。
- 如果脚本找不到模块，确认 **Working directory** 是 `$ProjectFileDir$`（确保 `import src.*` 能找到包）。
- 若 `webrtcvad` 等安装有问题，可先不启用 VAD；本仓库基线无需 VAD 也能跑通。
