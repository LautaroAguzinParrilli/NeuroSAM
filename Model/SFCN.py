"""
SFCN - Simple Fully Convolutional Network for Brain Age Prediction
Based on: Peng et al. (2021) "Accurate brain age prediction with lightweight deep neural networks"
Medical Image Analysis, 68, 101871.

Input:  (B, 1, 176, 208, 176)  — z-score normalised T1 MRI
Output: (B, 1)                 — predicted age (years)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv3d → BatchNorm3d → ReLU (standard SFCN block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, max_pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if max_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# SFCN
# ---------------------------------------------------------------------------

class SFCN(nn.Module):
    """
    Simple Fully Convolutional Network (Peng et al. 2021).

    Architecture (channel progression):  1 → 32 → 64 → 128 → 256 → 256 → 64
    Each of the first 5 blocks: Conv3d + BN + ReLU + MaxPool2
    Block 6:  1×1 Conv + BN + ReLU  (no pooling — acts as channel bottleneck)
    Head:     AdaptiveAvgPool → Dropout → FC → scalar

    The final linear layer produces a single regression value (age in years).
    """

    channel_config = [32, 64, 128, 256, 256, 64]

    def __init__(self, in_channels: int = 1, dropout: float = 0.5):
        super().__init__()

        ch = self.channel_config

        # ---- Convolutional backbone ----------------------------------------
        self.block1 = ConvBnRelu(in_channels, ch[0], kernel=3, padding=1, max_pool=True)
        self.block2 = ConvBnRelu(ch[0],       ch[1], kernel=3, padding=1, max_pool=True)
        self.block3 = ConvBnRelu(ch[1],       ch[2], kernel=3, padding=1, max_pool=True)
        self.block4 = ConvBnRelu(ch[2],       ch[3], kernel=3, padding=1, max_pool=True)
        self.block5 = ConvBnRelu(ch[3],       ch[4], kernel=3, padding=1, max_pool=True)

        # 1×1 bottleneck — no pooling
        self.block6 = ConvBnRelu(ch[4], ch[5], kernel=1, padding=0, max_pool=False)

        # ---- Regression head -----------------------------------------------
        self.pool    = nn.AdaptiveAvgPool3d(1)   # → (B, 64, 1, 1, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(ch[5], 1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W)  — single-channel T1 volume
        Returns:
            pred: (B, 1)         — predicted age
        """
        x = self.block1(x)   # → (B, 32,  D/2,  H/2,  W/2)
        x = self.block2(x)   # → (B, 64,  D/4,  H/4,  W/4)
        x = self.block3(x)   # → (B, 128, D/8,  H/8,  W/8)
        x = self.block4(x)   # → (B, 256, D/16, H/16, W/16)
        x = self.block5(x)   # → (B, 256, D/32, H/32, W/32)
        x = self.block6(x)   # → (B, 64,  D/32, H/32, W/32)

        x = self.pool(x)               # → (B, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)      # → (B, 64)
        x = self.dropout(x)
        x = self.fc(x)                 # → (B, 1)
        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SFCN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable:,}")

    dummy = torch.randn(2, 1, 176, 208, 176)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nInput shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}   ← (batch, 1) predicted ages")
    print(f"Sample preds: {out.squeeze().tolist()}")