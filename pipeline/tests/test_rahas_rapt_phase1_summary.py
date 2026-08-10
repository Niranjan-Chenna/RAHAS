from __future__ import annotations

import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_rahas_rapt_phase1_v1 import (
    DATASET_SHA256,
    METRICS,
    SEEDS,
    co_primary_verdict,
    generate_cross_model_qualitative_sheets,
    metrics_from_rows,
    paired_metric_statistics,
    require_exact_model_alignment,
    validate_prediction_rows,
)


LABELS = ("A", "B", "C", "D", "E")
COUNTS = (1, 2, 5, 10, 20)
BINS = ("one_shot", "2_4", "5_9", "10_19", "20_plus")
SCORES = (0.50, 0.20, 0.15, 0.10, 0.05)


def ranked(predicted: str) -> list[str]:
    return [predicted, *(label for label in LABELS if label != predicted)]


def make_rapt_rows(split: str = "test", paths: list[str] | None = None) -> list[dict[str, object]]:
    paths = paths or [f"images/{split}_{index}.png" for index in range(5)]
    rows = []
    for index, (label, count, frequency_bin, path) in enumerate(
        zip(LABELS, COUNTS, BINS, paths), start=1
    ):
        predicted_index = index if index != 2 else 1
        predicted = LABELS[predicted_index - 1]
        row: dict[str, object] = {
            "sample_path": path,
            "sample_id": f"sample-{index}",
            "split": split,
            "true_class_index": index,
            "predicted_class_index": predicted_index,
            "true_full_label": label,
            "predicted_full_label": predicted,
            "true_base_label": f"base-{label}",
            "predicted_base_from_full": f"base-{predicted}",
            "predicted_base_label": f"aux-base-{label}",
            "true_modifier_label": f"modifier-{label}",
            "predicted_modifier_from_full": f"modifier-{predicted}",
            "predicted_modifier_label": f"aux-modifier-{label}",
            "training_original_sample_count": count,
            "frequency_bin": frequency_bin,
            "correct": str(index == predicted_index).lower(),
            "model_seed": 2026,
            "checkpoint_sha256": "a" * 64,
            "dataset_sha256": DATASET_SHA256,
        }
        for rank, (top_label, score) in enumerate(zip(ranked(predicted), SCORES), start=1):
            row[f"top_{rank}_label"] = top_label
            row[f"top_{rank}_score"] = score
        rows.append(row)
    return rows


def make_resnet_rows(
    split: str = "test", paths: list[str] | None = None
) -> list[dict[str, object]]:
    rows = []
    for rapt in make_rapt_rows(split, paths):
        row: dict[str, object] = {
            "sample_path": rapt["sample_path"],
            "sample_id": rapt["sample_id"],
            "split": split,
            "class_index": rapt["true_class_index"],
            "predicted_class_index": rapt["predicted_class_index"],
            "class_label": rapt["true_full_label"],
            "predicted_label": rapt["predicted_full_label"],
            "true_base": rapt["true_base_label"],
            "predicted_base_from_character": rapt["predicted_base_from_full"],
            "true_modifier": rapt["true_modifier_label"],
            "predicted_modifier_from_character": rapt["predicted_modifier_from_full"],
            "training_original_count": rapt["training_original_sample_count"],
            "frequency_bin": rapt["frequency_bin"],
            "correct": rapt["correct"],
            "model_seed": 2026,
            "checkpoint_sha256": "b" * 64,
            "dataset_sha256": DATASET_SHA256,
        }
        for rank in range(1, 6):
            row[f"top_{rank}_label"] = rapt[f"top_{rank}_label"]
            row[f"top_{rank}_score"] = rapt[f"top_{rank}_score"]
        rows.append(row)
    return rows


class PredictionValidationTests(unittest.TestCase):
    def test_rejects_bad_provenance_and_non_monotonic_top5(self) -> None:
        rows = make_rapt_rows()
        rows[0]["checkpoint_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_prediction_rows(
                rows,
                True,
                context="rapt",
                seed=2026,
                split="test",
                checkpoint_hash="a" * 64,
            )

        rows = make_rapt_rows()
        rows[0]["top_3_score"] = 0.30
        with self.assertRaisesRegex(ValueError, "monotonically"):
            validate_prediction_rows(rows, True, context="rapt")

        rows = make_rapt_rows()
        rows[0]["top_5_score"] = "nan"
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction_rows(rows, True, context="rapt")

    def test_requires_exact_sample_and_target_alignment(self) -> None:
        rapt = make_rapt_rows()
        resnet = make_resnet_rows()
        require_exact_model_alignment(rapt, resnet, context="test")
        resnet[2]["class_label"] = "different"
        with self.assertRaisesRegex(ValueError, "sample or target mismatch"):
            require_exact_model_alignment(rapt, resnet, context="test")


class MetricTests(unittest.TestCase):
    def test_complete_metrics_use_full_label_components_and_separate_auxiliary(self) -> None:
        metrics = metrics_from_rows(make_rapt_rows(), True)
        self.assertTrue(set(METRICS).issubset(metrics))
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertEqual(metrics["base_accuracy"], 0.8)
        self.assertEqual(metrics["modifier_accuracy"], 0.8)
        self.assertEqual(metrics["auxiliary_base_accuracy"], 0.0)
        self.assertEqual(metrics["auxiliary_modifier_accuracy"], 0.0)
        self.assertEqual(metrics["top5"], 1.0)
        self.assertEqual(metrics["one_shot_accuracy"], 1.0)
        self.assertEqual(metrics["2_4_accuracy"], 0.0)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("weighted_f1", metrics)

        resnet = metrics_from_rows(make_resnet_rows(), False)
        self.assertEqual(resnet["base_accuracy"], metrics["base_accuracy"])
        self.assertNotIn("auxiliary_base_accuracy", resnet)

    def test_frequency_metrics_are_required_not_defaulted(self) -> None:
        rows = make_rapt_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "frequency bins have no samples"):
            metrics_from_rows(rows, True)


class PairedStatisticsTests(unittest.TestCase):
    @staticmethod
    def paired_rows(differences: list[float], metric: str) -> list[dict[str, object]]:
        return [
            {"seed": seed, f"difference_{metric}": difference}
            for seed, difference in zip(SEEDS, differences)
        ]

    def test_reports_low_n_t4_rank_and_median_distance(self) -> None:
        rows = self.paired_rows([0.10, 0.08, 0.06, 0.04, 0.02], "accuracy")
        result = paired_metric_statistics("accuracy", rows, bootstrap_samples=2000)
        self.assertEqual(result["paired_t_df"], 4)
        self.assertIn("LOW-N", result["low_n_warning"])
        self.assertEqual(result["seed2026_descending_rank"], 1.0)
        self.assertAlmostEqual(result["median_difference"], 0.06)
        self.assertAlmostEqual(result["seed2026_signed_distance_from_median"], 0.04)
        self.assertGreater(result["paired_t_ci95_low"], 0.0)

    def test_co_primary_verdict_is_conservative_when_one_endpoint_is_uncertain(self) -> None:
        supported_rows = self.paired_rows([0.10, 0.08, 0.06, 0.04, 0.02], "accuracy")
        accuracy = paired_metric_statistics("accuracy", supported_rows, bootstrap_samples=1000)
        macro_rows = self.paired_rows([0.02, -0.02, 0.01, -0.01, 0.00], "macro_f1")
        macro = paired_metric_statistics("macro_f1", macro_rows, bootstrap_samples=1000)
        decision = co_primary_verdict({"accuracy": accuracy, "macro_f1": macro})
        self.assertFalse(decision["recommend_rapt_as_primary"])
        self.assertIn("Uncertain", decision["verdict"])
        self.assertIn("Do not recommend", decision["recommendation"])


class QualitativeSheetTests(unittest.TestCase):
    def test_generates_cross_model_sheets_from_saved_prediction_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(5):
                path = root / f"sample_{index}.png"
                Image.new("L", (24, 24), color=40 * index).save(path)
                paths.append(str(path))
            rapt = make_rapt_rows(paths=paths)
            resnet = make_resnet_rows(paths=paths)
            resnet[0]["predicted_class_index"] = 2
            resnet[0]["predicted_label"] = "B"
            resnet[0]["top_1_label"] = "B"
            resnet[0]["top_2_label"] = "A"
            resnet[0]["correct"] = "false"
            output = root / "sheets"
            index = generate_cross_model_qualitative_sheets(
                rapt, resnet, output, seed=2026, page_size=2
            )
            self.assertEqual(len(index), 5)
            self.assertTrue((output / "rapt_correct_resnet_wrong_001.png").is_file())
            self.assertTrue((output / "cross_model_qualitative_index.csv").is_file())
            with (output / "cross_model_qualitative_index.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertIn("rapt_correct_resnet_wrong", {row["comparison"] for row in rows})


if __name__ == "__main__":
    unittest.main()
