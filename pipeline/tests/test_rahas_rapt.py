from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.ocr.rahas_rapt import (
    RahasRAPT,
    aggregate_rapt_prototypes,
    build_rahas_rapt_from_checkpoint,
    reliability_transport_logits,
    restoration_reliability_target,
    shot_aware_fusion_logits,
    shot_aware_query_router_logits,
)
from src.ocr.soft_data import OCRImageRecord, continuous_stroke_attenuation
from src.ocr.comparison_models import TorchvisionClassifier
from train_rahas_rapt_v1 import (
    SourceAwareEpisodeSampler,
    load_resnet_classifier_initialization,
    parse_args,
)


class RahasRAPTTests(unittest.TestCase):
    def build_model(self) -> RahasRAPT:
        return RahasRAPT(
            num_full_labels=9,
            num_base_glyphs=5,
            num_modifiers=3,
            token_dim=16,
            embedding_dim=24,
            grid_size=3,
            pretrained_backbone=False,
        )

    def test_forward_exposes_spatial_reliability_and_heads(self) -> None:
        model = self.build_model().eval()
        output = model(torch.rand(2, 5, 64, 64))
        self.assertEqual(tuple(output["tokens"].shape), (2, 9, 16))
        self.assertEqual(tuple(output["reliability"].shape), (2, 9))
        self.assertEqual(tuple(output["embedding"].shape), (2, 24))
        self.assertEqual(tuple(output["full_label"].shape), (2, 9))
        self.assertEqual(tuple(output["direct_full_label"].shape), (2, 9))
        self.assertTrue(bool(((output["reliability"] >= 0) & (output["reliability"] <= 1)).all()))

    def test_reliable_support_is_not_replaced_by_memory(self) -> None:
        model = self.build_model()
        tokens = torch.nn.functional.normalize(torch.rand(2, 9, 16), dim=-1)
        reliability = torch.ones(2, 9)
        completed, effective, gate = model.complete_support(
            tokens,
            reliability,
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
        )
        self.assertTrue(torch.allclose(completed, tokens, atol=1e-6))
        self.assertTrue(torch.allclose(effective, reliability, atol=1e-6))
        self.assertTrue(torch.allclose(gate, torch.zeros_like(gate), atol=1e-6))

    def test_disabled_prototype_completion_returns_support_unchanged(self) -> None:
        model = RahasRAPT(
            num_full_labels=9,
            num_base_glyphs=5,
            num_modifiers=3,
            token_dim=16,
            embedding_dim=24,
            grid_size=3,
            pretrained_backbone=False,
            prototype_completion=False,
        )
        tokens = torch.rand(2, 9, 16)
        reliability = torch.rand(2, 9)
        completed, effective, gate = model.complete_support(
            tokens,
            reliability,
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
        )
        self.assertIs(completed, tokens)
        self.assertIs(effective, reliability)
        self.assertTrue(torch.equal(gate, torch.zeros_like(reliability)))

    def test_prototype_completion_config_is_checkpointed_and_backward_compatible(self) -> None:
        enabled = self.build_model()
        self.assertTrue(enabled.checkpoint_config()["prototype_completion"])

        legacy_config = enabled.checkpoint_config()
        legacy_config.pop("prototype_completion")
        restored_legacy = build_rahas_rapt_from_checkpoint(
            {"model_config": legacy_config, "model_state": enabled.state_dict()}
        )
        self.assertTrue(restored_legacy.config.prototype_completion)

        disabled_config = {**legacy_config, "prototype_completion": False}
        restored_disabled = build_rahas_rapt_from_checkpoint(
            {"model_config": disabled_config, "model_state": enabled.state_dict()}
        )
        self.assertFalse(restored_disabled.config.prototype_completion)

    def test_disable_prototype_completion_cli_flag(self) -> None:
        with patch.object(sys, "argv", ["train_rahas_rapt_v1.py"]):
            self.assertFalse(parse_args().disable_prototype_completion)
        with patch.object(
            sys,
            "argv",
            ["train_rahas_rapt_v1.py", "--disable-prototype-completion"],
        ):
            self.assertTrue(parse_args().disable_prototype_completion)

    def test_episode_transport_has_expected_shape_and_gradients(self) -> None:
        model = self.build_model()
        output = model(torch.rand(6, 5, 64, 64))
        support = torch.tensor([True, True, True, False, False, False])
        local = torch.tensor([0, 1, 2, 0, 1, 2])
        prototype_tokens, prototype_reliability, prototype_embeddings = aggregate_rapt_prototypes(
            model,
            output["tokens"][support],
            output["reliability"][support],
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 1, 2]),
            local[support],
        )
        query_embeddings = model.tokens_to_embedding(
            output["tokens"][~support], output["reliability"][~support]
        )
        logits = reliability_transport_logits(
            model,
            output["tokens"][~support],
            output["reliability"][~support],
            query_embeddings,
            prototype_tokens,
            prototype_reliability,
            prototype_embeddings,
        )
        self.assertEqual(tuple(logits.shape), (3, 3))
        logits.sum().backward()
        self.assertIsNotNone(model.base_memory.grad)

    def test_reliability_target_and_fading_remain_continuous(self) -> None:
        features = torch.rand(5, 32, 32)
        faded = continuous_stroke_attenuation(features, random.Random(2026))
        target = restoration_reliability_target(faded.unsqueeze(0), 4)
        self.assertEqual(tuple(faded.shape), tuple(features.shape))
        self.assertEqual(tuple(target.shape), (1, 16))
        self.assertTrue(bool(((faded >= 0) & (faded <= 1)).all()))
        self.assertGreater(torch.unique(faded[1]).numel(), 20)

    def test_sampler_separates_augmentation_parents_when_available(self) -> None:
        records = []
        for class_index in range(1, 4):
            for parent_index in range(4):
                records.append(
                    OCRImageRecord(
                        path=Path(f"{class_index}_{parent_index}.png"),
                        label=str(class_index),
                        full_idx=class_index,
                        base_idx=class_index,
                        modifier_idx=0,
                        nasal=0.0,
                        original_crop_id=f"crop_{class_index}_{parent_index}",
                        augmentation_parent_id=f"parent_{class_index}_{parent_index}",
                    )
                )
        sampler = SourceAwareEpisodeSampler(records, 1, 3, (2,), 1, 2026)
        episode = next(iter(sampler))
        for local_class in range(3):
            selected = [item[0] for item in episode if item[2] == local_class]
            parents = {records[index].augmentation_parent_id for index in selected}
            self.assertEqual(len(parents), 3)

    def test_shot_router_decreases_transport_weight_with_class_frequency(self) -> None:
        transport = torch.randn(2, 3)
        direct = torch.randn(2, 4)
        labels = torch.tensor([1, 2, 3])
        counts = torch.tensor([0.0, 1.0, 4.0, 16.0])
        fused, weights = shot_aware_fusion_logits(
            transport, direct, labels, counts, transition_shots=4.0, slope=1.5
        )
        self.assertEqual(tuple(fused.shape), (2, 3))
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))
        reduced_fused, _ = shot_aware_fusion_logits(
            transport, direct[:, labels], labels, counts, transition_shots=4.0, slope=1.5
        )
        self.assertEqual(tuple(reduced_fused.shape), (2, 3))
        _, floored_weights = shot_aware_fusion_logits(
            transport,
            direct,
            labels,
            counts,
            transition_shots=1.5,
            slope=3.0,
            one_shot_floor=1.0,
            low_shot_floor=0.75,
        )
        self.assertEqual(float(floored_weights[0]), 1.0)
        self.assertGreaterEqual(float(floored_weights[1]), 0.75)

    def test_resnet_classifier_initialization_maps_backbone_and_class_rows(self) -> None:
        model = RahasRAPT(373, 5, 3, token_dim=16, embedding_dim=24, grid_size=3, pretrained_backbone=False)
        source = TorchvisionClassifier("resnet18", False, 372)
        labels = ["_", *[str(index) for index in range(372)]]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resnet.pt"
            torch.save(
                {
                    "epoch": 7,
                    "model_state": source.state_dict(),
                    "label_maps": {"idx_to_full_label": labels},
                },
                checkpoint,
            )
            report = load_resnet_classifier_initialization(
                model,
                checkpoint,
                {"idx_to_full_label": labels},
            )
        self.assertEqual(report["classifier_rows_copied"], 372)
        self.assertTrue(torch.allclose(model.direct_head.weight[1:], source.model.fc.weight))
        self.assertTrue(torch.allclose(model.backbone.stem[0].weight, source.model.conv1.weight))
        model.eval()
        source.eval()
        features = torch.rand(2, 5, 64, 64)
        rgb = features[:, :1].expand(-1, 3, -1, -1)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.inference_mode():
            expected = source((rgb - mean) / std)["full_label"]
            actual = model(features)["direct_full_label"][:, 1:]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_query_router_uses_transport_only_for_confident_scarce_prediction(self) -> None:
        transport = torch.tensor([[5.0, 1.0, 0.0], [0.0, 1.0, 5.0]])
        direct = torch.tensor([[0.0, 4.0, 1.0], [4.0, 1.0, 0.0]])
        labels = torch.tensor([1, 2, 3])
        counts = torch.tensor([0.0, 1.0, 4.0, 16.0])
        routed, use_transport, margins = shot_aware_query_router_logits(
            transport,
            direct,
            labels,
            counts,
            max_transport_shots=4,
            minimum_transport_margin=0.5,
        )
        self.assertEqual(use_transport.tolist(), [True, False])
        self.assertTrue(torch.all(margins > 0.5))
        self.assertTrue(torch.allclose(routed[0], torch.log_softmax(transport[0], dim=0)))
        self.assertTrue(torch.allclose(routed[1], torch.log_softmax(direct[1], dim=0)))


if __name__ == "__main__":
    unittest.main()
