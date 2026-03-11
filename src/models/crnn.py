# src/models/crnn.py
import torch
import torch.nn as nn


class CRNNClassifier(nn.Module):
    """
    输入: (B, T, F) 的 log-mel
    内部转成 (B, 1, F, T) 做 2D CNN
    只在频率维做池化（保持时间分辨率）
    """
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
        cnn_channels=(16, 32, 64),
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = int(n_mels)
        self.num_classes = int(num_classes)

        layers = []
        in_ch = 1
        freq = self.n_mels

        for out_ch in cnn_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # 只对 freq 维池化 (freq/2)，time 不动
                nn.MaxPool2d(kernel_size=(2, 1)),
            ]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            freq = (freq + 1) // 2  # 近似跟踪池化后的 freq

        self.cnn = nn.Sequential(*layers)

        rnn_in = in_ch * freq
        self.rnn = nn.GRU(
            input_size=rnn_in,
            hidden_size=int(rnn_hidden),
            num_layers=int(rnn_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(rnn_layers) > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(int(rnn_hidden) * 2, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,1,F,T)
        x = x.transpose(1, 2).unsqueeze(1)

        z = self.cnn(x)                # (B,C,F',T)
        z = z.permute(0, 3, 1, 2)      # (B,T,C,F')
        z = z.reshape(z.size(0), z.size(1), -1)  # (B,T,C*F')

        out, _ = self.rnn(z)           # (B,T,2H)
        pooled = out.mean(dim=1)       # clip-level mean pooling
        logits = self.head(pooled)     # (B,num_classes)
        return logits
