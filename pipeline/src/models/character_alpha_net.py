from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def activation_layer(name: str = "gelu") -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            group_norm(channels),
            activation_layer(activation),
            nn.Conv2d(channels, channels, 3, padding=1),
            group_norm(channels),
        )
        self.activation = activation_layer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            group_norm(out_channels),
            activation_layer(activation),
            ResidualBlock(out_channels, activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            group_norm(out_channels),
            activation_layer(activation),
            ResidualBlock(out_channels, activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, activation: str = "gelu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, activation)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CharacterAlphaNet(nn.Module):
    """Predict a soft character/matra alpha mask from noisy grayscale context."""

    def __init__(self, in_channels: int = 3, base_channels: int = 24, activation: str = "gelu") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.enc1 = ConvBlock(in_channels, base_channels, activation)
        self.enc2 = DownBlock(base_channels, base_channels * 2, activation)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4, activation)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8, activation)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, activation),
            ResidualBlock(base_channels * 8, activation),
        )
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, activation)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, activation)
        self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels, activation)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            activation_layer(activation),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"CharacterAlphaNet expected {self.in_channels} input channels, got {x.shape[1]}")
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)
