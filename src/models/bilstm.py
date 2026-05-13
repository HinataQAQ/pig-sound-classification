import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 4),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, 2H)
        mean = out.mean(dim=1)         # (B, 2H)
        mx = out.max(dim=1).values     # (B, 2H)
        h = torch.cat([mean, mx], dim=1)  # (B, 4H)
        logits = self.classifier(h)
        return logits
