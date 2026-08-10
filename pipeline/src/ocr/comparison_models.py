from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from src.ocr.rahas_spatial_proto import ResidualBlock


@dataclass(frozen=True)
class ComparisonProtoConfig:
    num_full_labels: int
    num_base_glyphs: int
    num_modifiers: int
    in_channels: int = 5
    base_channels: int = 16
    embedding_dim: int = 128
    grid_size: int = 6
    use_coordinates: bool = True
    preserve_spatial_grid: bool = True
    auxiliary_heads: bool = True


class ComparisonProtoNet(nn.Module):
    """Controlled variants of the RAHAS spatial prototypical recognizer."""

    def __init__(
        self,
        num_full_labels: int,
        num_base_glyphs: int,
        num_modifiers: int,
        in_channels: int = 5,
        base_channels: int = 16,
        embedding_dim: int = 128,
        grid_size: int = 6,
        use_coordinates: bool = True,
        preserve_spatial_grid: bool = True,
        auxiliary_heads: bool = True,
    ) -> None:
        super().__init__()
        self.config = ComparisonProtoConfig(
            num_full_labels=num_full_labels,
            num_base_glyphs=num_base_glyphs,
            num_modifiers=num_modifiers,
            in_channels=in_channels,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            grid_size=grid_size,
            use_coordinates=use_coordinates,
            preserve_spatial_grid=preserve_spatial_grid,
            auxiliary_heads=auxiliary_heads,
        )
        c = base_channels
        encoder_input = in_channels + (2 if use_coordinates else 0)
        self.encoder = nn.Sequential(
            ResidualBlock(encoder_input, c),
            ResidualBlock(c, c * 2, 2),
            ResidualBlock(c * 2, c * 3, 2),
            ResidualBlock(c * 3, c * 4, 2),
        )
        projection_input = c * 4 * grid_size * grid_size if preserve_spatial_grid else c * 4
        self.projection = nn.Sequential(
            nn.LayerNorm(projection_input),
            nn.Linear(projection_input, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.full_head = nn.Linear(embedding_dim, num_full_labels)
        self.base_head = nn.Linear(embedding_dim, num_base_glyphs)
        self.modifier_head = nn.Linear(embedding_dim, num_modifiers)
        self.nasal_head = nn.Linear(embedding_dim, 1)

    @staticmethod
    def _coordinates(x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        yy = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        y, x_coord = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((x_coord, y)).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(f"expected Bx{self.config.in_channels}xHxW input, got {tuple(x.shape)}")
        encoder_input = torch.cat((x, self._coordinates(x)), dim=1) if self.config.use_coordinates else x
        features = self.encoder(encoder_input)
        if self.config.preserve_spatial_grid:
            pooled = F.adaptive_avg_pool2d(features, self.config.grid_size).flatten(1)
        else:
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        embedding = F.normalize(self.projection(pooled), dim=1)
        return {
            "embedding": embedding,
            "full_label": self.full_head(embedding),
            "base_glyph": self.base_head(embedding),
            "modifier": self.modifier_head(embedding),
            "nasal": self.nasal_head(embedding).squeeze(1),
        }

    def checkpoint_config(self) -> dict:
        return asdict(self.config)


class PlainProtoNet(nn.Module):
    """Ordinary global-pooled prototypical network without RAHAS evidence or heads."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, embedding_dim: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, c),
            ResidualBlock(c, c * 2, 2),
            ResidualBlock(c * 2, c * 3, 2),
            ResidualBlock(c * 3, c * 4, 2),
        )
        self.projection = nn.Sequential(
            nn.Linear(c * 4, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)
        return {"embedding": F.normalize(self.projection(pooled), dim=1)}

    def checkpoint_config(self) -> dict:
        return {"in_channels": 1, "base_channels": 16, "embedding_dim": self.embedding_dim}


class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 372) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 96, 2),
            ResidualBlock(96, 128, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return {"full_label": self.classifier(pooled)}

    def checkpoint_config(self) -> dict:
        return {"architecture": "simple_cnn", "in_channels": 1, "num_classes": 372}


class TorchvisionClassifier(nn.Module):
    def __init__(self, architecture: str, pretrained: bool, num_classes: int = 372) -> None:
        super().__init__()
        if architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif architecture == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = efficientnet_b0(weights=weights)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported torchvision architecture: {architecture}")
        self.architecture = architecture
        self.pretrained = pretrained

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"full_label": self.model(x)}

    def checkpoint_config(self) -> dict:
        return {
            "architecture": self.architecture,
            "pretrained": self.pretrained,
            "in_channels": 3,
            "num_classes": 372,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
