import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)              # (B, T, 2H)
        mean = out.mean(dim=1)
        mx, _ = out.max(dim=1)
        h = torch.cat([mean, mx], dim=1)   # (B, 4H)
        proj = nn.Linear(h.shape[1], out.shape[2]).to(h.device)
        h = proj(h)                        # (B, 2H)
        logits = self.classifier(h)
        return logits
