"""Final OCR components for RAHAS."""

from .label_decomposition import (
    NONE,
    LabelRecord,
    build_label_maps,
    list_character_labels,
    parse_label,
)
from .rahas_spatial_proto import (
    RahasSpatialProto,
    SpatialProtoConfig,
    build_rahas_spatial_proto_from_checkpoint,
    squared_euclidean_logits,
)

__all__ = [
    "NONE",
    "LabelRecord",
    "build_label_maps",
    "build_rahas_spatial_proto_from_checkpoint",
    "list_character_labels",
    "parse_label",
    "RahasSpatialProto",
    "SpatialProtoConfig",
    "squared_euclidean_logits",
]
