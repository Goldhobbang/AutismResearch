"""PyTorch Deep Learning Time-Series Models for WESAD Stress Detection.
Includes 1D-CNN, TCN (Temporal Convolutional Network), and BiGRU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, pool_size=2, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class CNN1DStressModel(nn.Module):
    """1D CNN Architecture for continuous physiological & motion time-series."""
    def __init__(self, in_channels=6, num_classes=2, dropout=0.3):
        super().__init__()
        # Input: (B, in_channels, 1920)
        self.features = nn.Sequential(
            Conv1DBlock(in_channels, 32, kernel_size=7, padding=3, pool_size=4),   # 1920 -> 480
            Conv1DBlock(32, 64, kernel_size=5, padding=2, pool_size=4),            # 480 -> 120
            Conv1DBlock(64, 128, kernel_size=5, padding=2, pool_size=4),           # 120 -> 30
            Conv1DBlock(128, 128, kernel_size=3, padding=1, pool_size=2),          # 30 -> 15
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat).squeeze(-1)
        return self.classifier(pooled)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNStressModel(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions."""
    def __init__(self, in_channels=6, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2, num_classes=2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1) * dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        y = self.network(x)
        pooled = self.global_pool(y).squeeze(-1)
        return self.classifier(pooled)


class GRUStressModel(nn.Module):
    """Bidirectional GRU with self-attention / adaptive pooling."""
    def __init__(self, in_channels=6, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        # Pre-downsampling conv to reduce sequence length from 1920 to 240 for GRU speed
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=8),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 6, 1920)
        h = self.downsample(x) # (B, 32, 240)
        h = h.transpose(1, 2)  # (B, 240, 32)
        out, _ = self.gru(h)   # (B, 240, hidden_size * 2)
        pooled = torch.mean(out, dim=1) # (B, hidden_size * 2)
        return self.fc(pooled)
