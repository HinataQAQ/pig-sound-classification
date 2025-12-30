import torch
import torch.nn as nn


class CRNNClassifier(nn.Module):
    """
    输入: x (B, T, F)  例如 (B, 97, 64)
    输出: logits (B, num_classes)
    """
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
        cnn_channels=(16, 32, 64),
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_ch = 1
        # 只在频率维做池化 (2,1)，尽量保留时间维给 RNN 学“事件发生过程”
        for out_ch in cnn_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),   # F: 64->32->16->8,  T: 不变
                nn.Dropout(p=dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # CNN 后：形状 (B, C, F', T)
        # 我们把 (C*F') 当作每个时间步的特征维度，送进 RNN
        f_after = n_mels
        for _ in cnn_channels:
            f_after = (f_after + 1) // 2   # MaxPool(2,1) 对频率维约减半
        self.feature_dim = cnn_channels[-1] * f_after

        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(rnn_hidden * 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,1,F,T)
        x = x.transpose(1, 2).unsqueeze(1)

        z = self.cnn(x)                # (B,C,F',T)
        z = z.permute(0, 3, 1, 2)      # (B,T,C,F')
        z = z.reshape(z.size(0), z.size(1), -1)  # (B,T,C*F')

        out, _ = self.rnn(z)           # (B,T,2H)
        pooled = 0.5 * (out.mean(dim=1) + out.max(dim=1).values)
        logits = self.head(pooled)     # (B,num_classes)
        return logits
