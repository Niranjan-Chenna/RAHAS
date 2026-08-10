from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_rahas_rapt_phase1_v1 import (
    classification_metrics,
    prediction_mode_logits,
    select_router,
)


class RAPTPhase1Tests(unittest.TestCase):
    def test_classification_metrics_include_requested_statistics(self):
        target = np.asarray([1, 1, 2, 2])
        prediction = np.asarray([1, 2, 2, 2])
        top = np.asarray([[1, 2, 3], [2, 1, 3], [2, 1, 3], [2, 3, 1]])
        metrics, per_class = classification_metrics(prediction, target, top, target)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["top3"], 1.0)
        self.assertEqual(len(per_class), 2)
        for field in ("macro_precision", "macro_recall", "macro_f1", "weighted_f1", "top5"):
            self.assertIn(field, metrics)

    def test_router_selection_uses_validation_bundle_only(self):
        bundle = {
            "transport": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
            "direct": torch.tensor([[0.0, 5.0], [5.0, 0.0]]),
            "target": torch.tensor([1, 2]),
            "prototype_labels": torch.tensor([1, 2]),
        }
        shots = torch.tensor([0.0, 1.0, 1.0])
        selected, candidates = select_router(bundle, shots)
        self.assertEqual(selected["validation_one_shot_accuracy"], 1.0)
        self.assertGreater(len(candidates), 1)

    def test_fixed_prediction_modes_do_not_select_router_parameters(self):
        bundle = {
            "transport": torch.tensor([[4.0, 1.0], [0.0, 3.0]]),
            "direct": torch.tensor([[1.0, 4.0], [3.0, 0.0]]),
            "prototype_labels": torch.tensor([1, 2]),
        }
        shots = torch.tensor([0.0, 1.0, 10.0])
        direct, direct_route, _, direct_fraction = prediction_mode_logits(
            bundle, shots, {}, "direct_only"
        )
        self.assertTrue(torch.allclose(direct, torch.log_softmax(bundle["direct"], dim=1)))
        self.assertFalse(bool(direct_route.any()))
        self.assertEqual(direct_fraction, 0.0)

        fused, fused_route, _, fused_fraction = prediction_mode_logits(
            bundle, shots, {}, "equal_fusion"
        )
        expected = 0.5 * (
            torch.log_softmax(bundle["transport"], dim=1)
            + torch.log_softmax(bundle["direct"], dim=1)
        )
        self.assertTrue(torch.allclose(fused, expected))
        self.assertFalse(bool(fused_route.any()))
        self.assertEqual(fused_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
