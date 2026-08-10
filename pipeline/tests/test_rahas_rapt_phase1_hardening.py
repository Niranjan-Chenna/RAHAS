from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_rahas_rapt_phase1_v1 import (
    classification_metrics,
    frequency_bin,
    immutable_output_directory,
    prediction_package,
)
from train_rahas_rapt_v1 import (
    checkpoint_provenance,
    class_map_sha256,
    validate_checkpoint_provenance,
    verify_frozen_split,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Phase1HardeningTests(unittest.TestCase):
    def test_classification_macro_uses_ground_truth_classes_only(self):
        prediction = np.asarray([3, 2])
        target = np.asarray([1, 2])

        metrics, rows = classification_metrics(prediction, target, None, None, [1, 2])

        self.assertEqual([row["class_index"] for row in rows], [1, 2])
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)

    def test_zero_shot_has_its_own_frequency_bin(self):
        self.assertEqual(frequency_bin(0), "zero_shot")
        self.assertEqual(frequency_bin(1), "one_shot")

    def test_checkpoint_provenance_enforces_seed_dataset_and_class_map(self):
        maps = {
            "idx_to_full_label": ["unknown", "A"],
            "idx_to_base_glyph": ["unknown", "base"],
            "idx_to_modifier": ["unknown", "none"],
        }
        frozen = {"dataset_sha256": "a" * 64, "manifest_sha256": {"train_manifest.csv": "b" * 64}}
        expected = checkpoint_provenance(17, frozen, maps)
        checkpoint = {
            "args": {"seed": 17},
            "dataset_sha256": "a" * 64,
            "label_maps": maps,
        }
        validated = validate_checkpoint_provenance(checkpoint, expected, maps, "test checkpoint")
        self.assertEqual(validated["class_map_sha256"], class_map_sha256(maps))

        for key, value in (("seed", 42), ("dataset", "c" * 64), ("class_map", ["unknown", "B"])):
            changed = {
                "args": dict(checkpoint["args"]),
                "dataset_sha256": checkpoint["dataset_sha256"],
                "label_maps": dict(maps),
            }
            if key == "seed":
                changed["args"]["seed"] = value
            elif key == "dataset":
                changed["dataset_sha256"] = value
            else:
                changed["label_maps"]["idx_to_full_label"] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint_provenance(changed, expected, maps, "test checkpoint")

    def test_frozen_split_hashes_are_verified_and_test_is_deferred(self):
        with tempfile.TemporaryDirectory() as temporary:
            split = Path(temporary)
            rows = [
                {"split": "train", "sample_path": "train.png", "file_sha256": "1" * 64},
                {"split": "validation", "sample_path": "validation.png", "file_sha256": "2" * 64},
                {"split": "test", "sample_path": "test.png", "file_sha256": "3" * 64},
            ]
            write_csv(split / "canonical_manifest.csv", rows)
            for name, row in (
                ("train_manifest.csv", rows[0]),
                ("validation_manifest.csv", rows[1]),
                ("test_manifest.csv", rows[2]),
                ("class_distribution.csv", {"class": "A", "count": "3"}),
            ):
                write_csv(split / name, [row])
            payload = "\n".join(
                f"{row['split']}\t{row['sample_path']}\t{row['file_sha256']}" for row in rows
            )
            names = (
                "canonical_manifest.csv",
                "train_manifest.csv",
                "validation_manifest.csv",
                "test_manifest.csv",
                "class_distribution.csv",
            )
            summary = {
                "seed": 2026,
                "dataset_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "manifest_sha256": {name: file_hash(split / name) for name in names},
            }
            (split / "split_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (split / "leakage_audit.json").write_text(
                json.dumps({"status": "PASS", "checks": {"all": True}}), encoding="utf-8"
            )

            partial = verify_frozen_split(split, include_test=False)
            self.assertFalse(partial["test_material_verified"])
            (split / "test_manifest.csv").write_text("tampered\n", encoding="utf-8")
            verify_frozen_split(split, include_test=False)
            with self.assertRaises(RuntimeError):
                verify_frozen_split(split, include_test=True)

    def test_immutable_output_is_published_once_and_cleans_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            with immutable_output_directory(output) as staging:
                (staging / "metrics.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue((output / "metrics.json").exists())
            with self.assertRaises(FileExistsError):
                with immutable_output_directory(output):
                    pass

            failed = Path(temporary) / "failed"
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with immutable_output_directory(failed) as staging:
                    (staging / "partial.txt").write_text("partial", encoding="utf-8")
                    raise RuntimeError("stop")
            self.assertFalse(failed.exists())

    def test_prediction_package_reports_auxiliary_full_and_zero_shot_metrics(self):
        labels = torch.tensor([1, 2])
        bundle = {
            "transport": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
            "direct": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
            "target": torch.tensor([1, 2]),
            "base": torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]),
            "modifier": torch.tensor([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]),
            "true_base": torch.tensor([1, 2]),
            "true_modifier": torch.tensor([1, 2]),
            "query_reliability": torch.tensor([0.8, 0.7]),
            "prototype_labels": labels,
            "prototype_completion": torch.tensor([0.1, 0.2]),
        }
        maps = {
            "idx_to_full_label": ["unknown", "A", "B"],
            "idx_to_base_glyph": ["unknown", "x", "y"],
            "idx_to_modifier": ["unknown", "m", "n"],
            "records": [
                {"full_idx": 1, "base_idx": 1, "modifier_idx": 1},
                {"full_idx": 2, "base_idx": 2, "modifier_idx": 2},
            ],
        }
        records = [
            SimpleNamespace(path=Path.cwd() / "a.png"),
            SimpleNamespace(path=Path.cwd() / "b.png"),
        ]
        manifest = {
            "a.png": {"sample_path": "a.png"},
            "b.png": {"sample_path": "b.png"},
        }
        _, metrics, _ = prediction_package(
            bundle,
            records,
            manifest,
            maps,
            {},
            torch.zeros(3),
            {"max_transport_shots": 9, "minimum_transport_margin": 0.0},
            "c" * 64,
            "test",
            17,
            "d" * 64,
        )
        self.assertEqual(metrics["zero_shot_samples"], 2)
        self.assertAlmostEqual(metrics["auxiliary_base_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["full_label_base_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["auxiliary_modifier_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["full_label_modifier_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
