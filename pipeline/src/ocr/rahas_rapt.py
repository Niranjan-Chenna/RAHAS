from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class RAPTConfig:
    num_full_labels: int
    num_base_glyphs: int
    num_modifiers: int
    in_channels: int = 5
    token_dim: int = 128
    embedding_dim: int = 256
    grid_size: int = 6
    pretrained_backbone: bool = True
    prototype_completion: bool = True


class ResNet18SpatialEncoder(nn.Module):
    """Expose the 6x6 and deep feature maps of a ResNet-18 backbone."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.register_buffer("imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = gray.expand(-1, 3, -1, -1)
        rgb = (rgb - self.imagenet_mean) / self.imagenet_std
        x = self.stem(rgb)
        x = self.layer1(x)
        x = self.layer2(x)
        spatial = self.layer3(x)
        deep = self.layer4(spatial)
        return spatial, deep


class RahasRAPT(nn.Module):
    """Restoration-Aware Prototype Transport for low-shot Brahmi OCR.

    The appearance encoder never receives hard masks. Continuous restoration
    evidence predicts token reliability and controls where compositional memory
    is allowed to repair a support representation.
    """

    def __init__(
        self,
        num_full_labels: int,
        num_base_glyphs: int,
        num_modifiers: int,
        in_channels: int = 5,
        token_dim: int = 128,
        embedding_dim: int = 256,
        grid_size: int = 6,
        pretrained_backbone: bool = True,
        prototype_completion: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 5:
            raise ValueError("RAHAS-RAPT expects gray, darkness, contrast, soft-alpha, and gradient channels")
        self.config = RAPTConfig(
            num_full_labels=num_full_labels,
            num_base_glyphs=num_base_glyphs,
            num_modifiers=num_modifiers,
            in_channels=in_channels,
            token_dim=token_dim,
            embedding_dim=embedding_dim,
            grid_size=grid_size,
            pretrained_backbone=pretrained_backbone,
            prototype_completion=prototype_completion,
        )
        self.backbone = ResNet18SpatialEncoder(pretrained_backbone)
        self.appearance_projection = nn.Conv2d(256, token_dim, 1, bias=False)
        self.evidence_adapter = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, token_dim, 1, bias=False),
        )
        self.reliability_head = nn.Sequential(
            nn.Conv2d(token_dim + 4, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(512 + token_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.prototype_projection = nn.Sequential(
            nn.Linear(token_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.full_head = nn.Linear(embedding_dim, num_full_labels)
        self.direct_head = nn.Linear(512, num_full_labels)
        self.base_head = nn.Linear(embedding_dim, num_base_glyphs)
        self.modifier_head = nn.Linear(embedding_dim, num_modifiers)
        self.nasal_head = nn.Linear(embedding_dim, 1)

        token_count = grid_size * grid_size
        self.base_memory = nn.Parameter(torch.empty(num_base_glyphs, token_count, token_dim))
        self.modifier_memory = nn.Parameter(torch.empty(num_modifiers, token_count, token_dim))
        self.base_presence = nn.Parameter(torch.full((num_base_glyphs, token_count), -1.5))
        self.modifier_presence = nn.Parameter(torch.full((num_modifiers, token_count), -2.0))
        nn.init.trunc_normal_(self.base_memory, std=0.02)
        nn.init.trunc_normal_(self.modifier_memory, std=0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(f"expected Bx{self.config.in_channels}xHxW input, got {tuple(x.shape)}")
        spatial, deep = self.backbone(x[:, :1])
        grid_size = self.config.grid_size
        appearance = F.adaptive_avg_pool2d(self.appearance_projection(spatial), grid_size)
        evidence = F.adaptive_avg_pool2d(x[:, 1:], grid_size)
        evidence_features = self.evidence_adapter(evidence)
        reliability_logits = self.reliability_head(torch.cat((appearance, evidence), dim=1))
        reliability = torch.sigmoid(reliability_logits).flatten(1)
        fused = appearance + reliability_logits.sigmoid() * evidence_features
        tokens = F.normalize(fused.flatten(2).transpose(1, 2), dim=-1)

        weighted_tokens = self._weighted_pool(tokens, reliability)
        deep_features = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        embedding = F.normalize(
            self.embedding_projection(torch.cat((deep_features, weighted_tokens), dim=1)),
            dim=1,
        )
        return {
            "embedding": embedding,
            "tokens": tokens,
            "reliability": reliability,
            "reliability_logits": reliability_logits.flatten(1),
            "full_label": self.full_head(embedding),
            "direct_full_label": self.direct_head(deep_features),
            "base_glyph": self.base_head(embedding),
            "modifier": self.modifier_head(embedding),
            "nasal": self.nasal_head(embedding).squeeze(1),
        }

    @staticmethod
    def _weighted_pool(tokens: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        weights = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    def structural_prior(
        self,
        base_indices: torch.Tensor,
        modifier_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior = F.normalize(
            self.base_memory[base_indices] + self.modifier_memory[modifier_indices],
            dim=-1,
        )
        presence = torch.sigmoid(
            self.base_presence[base_indices] + self.modifier_presence[modifier_indices]
        )
        return prior, presence

    def complete_support(
        self,
        support_tokens: torch.Tensor,
        reliability: torch.Tensor,
        base_indices: torch.Tensor,
        modifier_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.prototype_completion:
            return support_tokens, reliability, torch.zeros_like(reliability)

        prior, prior_presence = self.structural_prior(base_indices, modifier_indices)
        attention = torch.softmax(
            torch.bmm(support_tokens, prior.transpose(1, 2)) / math.sqrt(self.config.token_dim),
            dim=-1,
        )
        aligned_prior = torch.bmm(attention, prior)
        aligned_presence = torch.bmm(attention, prior_presence.unsqueeze(-1)).squeeze(-1)
        repair_gate = (1.0 - reliability) * aligned_presence
        completed = F.normalize(
            (1.0 - repair_gate.unsqueeze(-1)) * support_tokens
            + repair_gate.unsqueeze(-1) * aligned_prior,
            dim=-1,
        )
        effective_reliability = reliability + (1.0 - reliability) * aligned_presence
        return completed, effective_reliability, repair_gate

    def tokens_to_embedding(self, tokens: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.prototype_projection(self._weighted_pool(tokens, reliability)), dim=1)

    def checkpoint_config(self) -> dict[str, int | bool]:
        return asdict(self.config)


def restoration_reliability_target(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Build a continuous stroke-evidence target without thresholding pixels."""

    if x.ndim != 4 or x.shape[1] != 5:
        raise ValueError(f"expected Bx5xHxW input, got {tuple(x.shape)}")
    darkness, contrast, soft_alpha, gradient = x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    evidence = 0.30 * darkness + 0.20 * contrast + 0.35 * soft_alpha + 0.15 * gradient
    evidence = evidence.unsqueeze(1).clamp(0.0, 1.0)
    average = F.adaptive_avg_pool2d(evidence, grid_size)
    maximum = F.adaptive_max_pool2d(evidence, grid_size)
    return (0.35 * average + 0.65 * maximum).flatten(1).clamp(0.0, 1.0)


def aggregate_rapt_prototypes(
    model: RahasRAPT,
    tokens: torch.Tensor,
    reliability: torch.Tensor,
    base_indices: torch.Tensor,
    modifier_indices: torch.Tensor,
    local_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    completed, effective_reliability, _ = model.complete_support(
        tokens,
        reliability,
        base_indices,
        modifier_indices,
    )
    class_count = int(local_labels.max().item()) + 1
    prototype_tokens = []
    prototype_reliability = []
    for class_index in range(class_count):
        mask = local_labels == class_index
        prototype_tokens.append(completed[mask].mean(dim=0))
        prototype_reliability.append(effective_reliability[mask].mean(dim=0))
    token_tensor = F.normalize(torch.stack(prototype_tokens), dim=-1)
    reliability_tensor = torch.stack(prototype_reliability).clamp(0.0, 1.0)
    embeddings = model.tokens_to_embedding(token_tensor, reliability_tensor)
    return token_tensor, reliability_tensor, embeddings


def reliability_transport_logits(
    model: RahasRAPT,
    query_tokens: torch.Tensor,
    query_reliability: torch.Tensor,
    query_embeddings: torch.Tensor,
    prototype_tokens: torch.Tensor,
    prototype_reliability: torch.Tensor,
    prototype_embeddings: torch.Tensor,
    match_temperature: float = 0.08,
    classification_temperature: float = 0.12,
) -> torch.Tensor:
    """Reciprocal soft token transport, weighted by restoration reliability."""

    similarities = torch.einsum("qtd,csd->qcts", query_tokens, prototype_tokens)
    query_match = match_temperature * torch.logsumexp(similarities / match_temperature, dim=-1)
    query_score = (
        query_match * query_reliability[:, None, :]
    ).sum(dim=-1) / query_reliability.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    prototype_match = match_temperature * torch.logsumexp(
        similarities.transpose(-1, -2) / match_temperature,
        dim=-1,
    )
    prototype_score = (
        prototype_match * prototype_reliability[None, :, :]
    ).sum(dim=-1) / prototype_reliability.sum(dim=-1).unsqueeze(0).clamp_min(1e-6)

    local_score = 0.5 * (query_score + prototype_score)
    global_score = query_embeddings @ prototype_embeddings.transpose(0, 1)
    return (0.65 * local_score + 0.35 * global_score) / classification_temperature


def shot_aware_fusion_logits(
    transport_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    prototype_labels: torch.Tensor,
    original_shot_counts: torch.Tensor,
    transition_shots: float = 6.0,
    slope: float = 1.5,
    one_shot_floor: float = 0.0,
    low_shot_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route scarce classes toward RAPT and frequent classes toward the direct head."""

    if original_shot_counts.ndim != 1:
        raise ValueError("original_shot_counts must be a one-dimensional class-count tensor")
    counts = original_shot_counts[prototype_labels].to(
        device=transport_logits.device,
        dtype=transport_logits.dtype,
    ).clamp_min(1.0)
    transport_weight = 1.0 / (1.0 + (counts / transition_shots) ** slope)
    floor = torch.where(
        counts <= 1.0,
        torch.full_like(counts, one_shot_floor),
        torch.where(counts <= 4.0, torch.full_like(counts, low_shot_floor), torch.zeros_like(counts)),
    )
    transport_weight = torch.maximum(transport_weight, floor)
    transport_log_prob = F.log_softmax(transport_logits, dim=1)
    if direct_logits.shape[1] == prototype_labels.numel():
        selected_direct_logits = direct_logits
    else:
        selected_direct_logits = direct_logits[:, prototype_labels]
    direct_log_prob = F.log_softmax(selected_direct_logits, dim=1)
    fused = (
        transport_weight.unsqueeze(0) * transport_log_prob
        + (1.0 - transport_weight).unsqueeze(0) * direct_log_prob
    )
    return fused, transport_weight


def shot_aware_query_router_logits(
    transport_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    prototype_labels: torch.Tensor,
    original_shot_counts: torch.Tensor,
    max_transport_shots: int = 4,
    minimum_transport_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one expert per query from RAPT confidence and predicted class frequency."""

    if direct_logits.shape[1] == prototype_labels.numel():
        selected_direct_logits = direct_logits
    else:
        selected_direct_logits = direct_logits[:, prototype_labels]
    transport_log_prob = F.log_softmax(transport_logits, dim=1)
    direct_log_prob = F.log_softmax(selected_direct_logits, dim=1)
    top_values, top_indices = transport_log_prob.topk(2, dim=1)
    margin = top_values[:, 0] - top_values[:, 1]
    predicted_labels = prototype_labels[top_indices[:, 0]]
    predicted_shots = original_shot_counts[predicted_labels].to(transport_logits.device)
    use_transport = (predicted_shots <= max_transport_shots) & (margin >= minimum_transport_margin)
    routed = torch.where(use_transport[:, None], transport_log_prob, direct_log_prob)
    return routed, use_transport, margin


def build_rahas_rapt_from_checkpoint(checkpoint: dict) -> RahasRAPT:
    model = RahasRAPT(**checkpoint["model_config"])
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    allowed_missing = {
        "direct_head.weight",
        "direct_head.bias",
        "backbone.imagenet_mean",
        "backbone.imagenet_std",
    }
    if unexpected or not missing.issubset(allowed_missing):
        raise RuntimeError(f"Incompatible RAPT checkpoint: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    return model
