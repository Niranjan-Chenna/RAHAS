from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            _norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.body(x) + self.skip(x))


@dataclass(frozen=True)
class SpatialProtoConfig:
    num_full_labels: int
    num_base_glyphs: int
    num_modifiers: int
    in_channels: int = 5
    base_channels: int = 24
    embedding_dim: int = 192
    grid_size: int = 6


class RahasSpatialProto(nn.Module):
    """Position-preserving metric OCR over continuous restoration evidence."""

    def __init__(
        self,
        num_full_labels: int,
        num_base_glyphs: int,
        num_modifiers: int,
        in_channels: int = 5,
        base_channels: int = 24,
        embedding_dim: int = 192,
        grid_size: int = 6,
    ) -> None:
        super().__init__()
        self.config = SpatialProtoConfig(
            num_full_labels=num_full_labels,
            num_base_glyphs=num_base_glyphs,
            num_modifiers=num_modifiers,
            in_channels=in_channels,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            grid_size=grid_size,
        )
        c = base_channels
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels + 2, c),
            ResidualBlock(c, c * 2, 2),
            ResidualBlock(c * 2, c * 3, 2),
            ResidualBlock(c * 3, c * 4, 2),
        )
        spatial_dim = c * 4 * grid_size * grid_size
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, embedding_dim * 2),
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
        features = self.encoder(torch.cat((x, self._coordinates(x)), dim=1))
        grid = F.adaptive_avg_pool2d(features, self.config.grid_size)
        embedding = F.normalize(self.spatial_projection(grid.flatten(1)), dim=1)
        return {
            "embedding": embedding,
            "full_label": self.full_head(embedding),
            "base_glyph": self.base_head(embedding),
            "modifier": self.modifier_head(embedding),
            "nasal": self.nasal_head(embedding).squeeze(1),
            "spatial_grid": grid,
        }

    def checkpoint_config(self) -> dict[str, int]:
        return asdict(self.config)


def squared_euclidean_logits(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    return -torch.cdist(queries, prototypes, p=2).square() / temperature


def build_rahas_spatial_proto_from_checkpoint(checkpoint: dict) -> RahasSpatialProto:
    model = RahasSpatialProto(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model
