from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.ocr.comparison_models import ComparisonProtoNet, PlainProtoNet, SimpleCNNClassifier
from src.ocr.soft_data import OCRImageRecord
from run_rahas_comparison_v1 import EpisodeSampler


class ComparisonModelTests(unittest.TestCase):
    def test_spatial_proto_ablation_shapes(self) -> None:
        sample = torch.zeros(2, 5, 96, 96)
        for coordinates, spatial in [(True, True), (False, True), (True, False)]:
            model = ComparisonProtoNet(
                373,
                73,
                10,
                in_channels=5,
                base_channels=4,
                embedding_dim=16,
                grid_size=3,
                use_coordinates=coordinates,
                preserve_spatial_grid=spatial,
            )
            output = model(sample)
            self.assertEqual(tuple(output["embedding"].shape), (2, 16))
            self.assertEqual(tuple(output["full_label"].shape), (2, 373))

    def test_plain_baselines_output_expected_shapes(self) -> None:
        proto = PlainProtoNet(1, 4, 16)(torch.zeros(2, 1, 96, 96))
        classifier = SimpleCNNClassifier(372)(torch.zeros(2, 1, 96, 96))
        self.assertEqual(tuple(proto["embedding"].shape), (2, 16))
        self.assertEqual(tuple(classifier["full_label"].shape), (2, 372))

    def test_natural_episode_sampler_is_deterministic(self) -> None:
        records = []
        for class_index, count in [(1, 2), (2, 4), (3, 8), (4, 16)]:
            records.extend(
                OCRImageRecord(Path(f"{class_index}_{index}.png"), str(class_index), class_index, 1, 1, 0.0)
                for index in range(count)
            )
        first = EpisodeSampler(records, 2, 3, 1, 1, 2026, "natural")
        second = EpisodeSampler(records, 2, 3, 1, 1, 2026, "natural")
        self.assertEqual(list(first), list(second))


if __name__ == "__main__":
    unittest.main()
