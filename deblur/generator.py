# Deblur generator architecture

import torch
import torch.nn as nn

# ============================================================
# Residual Block (EDSR Style)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.PReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x) * self.residual_scale

# ============================================================
# Deblur Generator
# ============================================================
class DeblurGenerator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super().__init__()
        # Initial Feature Extraction
        self.initial = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=9,
                stride=1,
                padding=4
            ),
            nn.PReLU()
        )

        # Residual Trunk
        self.residuals = nn.Sequential(
            *[
                ResidualBlock(64)
                for _ in range(num_residual_blocks)
            ]
        )

        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64)
        )

        # Reconstruction
        self.reconstruction = nn.Conv2d(
            64,
            3,
            kernel_size=9,
            stride=1,
            padding=4
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.fusion(x)
        x = x + initial
        x = self.reconstruction(x)
        return x
