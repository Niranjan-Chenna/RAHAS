from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_rahas_rapt_phase1_v1 import (
    DATASET_SHA256,
    METRICS,
    SEEDS,
    SEED_ARTIFACTS,
    metrics_from_predictions,
    verify_phase1,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("placeholder\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rapt_rows(split: str) -> list[dict[str, object]]:
    paths = [f"images/{split}_a.png", f"images/{split}_b.png"]
    return [
        {
            "sample_path": paths[0], "split": split,
            "true_class_index": 1, "predicted_class_index": 1,
            "true_full_label": "A", "predicted_full_label": "A",
            "true_base_label": "base-a", "predicted_base_label": "base-a",
            "predicted_base_from_full": "base-a",
            "true_modifier_label": "none", "predicted_modifier_label": "none",
            "predicted_modifier_from_full": "none",
            "frequency_bin": "one_shot",
            "top_1_label": "A", "top_2_label": "B", "top_3_label": "C",
            "top_4_label": "D", "top_5_label": "E",
        },
        {
            "sample_path": paths[1], "split": split,
            "true_class_index": 2, "predicted_class_index": 2,
            "true_full_label": "B", "predicted_full_label": "B",
            "true_base_label": "base-b", "predicted_base_label": "base-b",
            "predicted_base_from_full": "base-b",
            "true_modifier_label": "mark", "predicted_modifier_label": "mark",
            "predicted_modifier_from_full": "mark",
            "frequency_bin": "one_shot",
            "top_1_label": "B", "top_2_label": "A", "top_3_label": "C",
            "top_4_label": "D", "top_5_label": "E",
        },
    ]


def resnet_rows(split: str) -> list[dict[str, object]]:
    paths = [f"images/{split}_a.png", f"images/{split}_b.png"]
    return [
        {
            "sample_path": paths[0], "split": split,
            "class_index": 1, "predicted_class_index": 1,
            "class_label": "A", "predicted_label": "A",
            "true_base": "base-a", "predicted_base": "base-a",
            "true_modifier": "none", "predicted_modifier": "none",
            "frequency_bin": "one_shot",
            "top_1_label": "A", "top_2_label": "B", "top_3_label": "C",
            "top_4_label": "D", "top_5_label": "E",
        },
        {
            "sample_path": paths[1], "split": split,
            "class_index": 2, "predicted_class_index": 1,
            "class_label": "B", "predicted_label": "A",
            "true_base": "base-b", "predicted_base": "base-a",
            "true_modifier": "mark", "predicted_modifier": "none",
            "frequency_bin": "one_shot",
            "top_1_label": "A", "top_2_label": "B", "top_3_label": "C",
            "top_4_label": "D", "top_5_label": "E",
        },
    ]


def build_package(root: Path) -> None:
    paired_rows: list[dict[str, object]] = []
    repeated_rows: list[dict[str, object]] = []
    prediction_dir = root / "seed_level_predictions"
    metric_dir = root / "seed_level_metrics"

    for seed in SEEDS:
        seed_dir = root / "runs" / f"seed_{seed}"
        rapt_dir = seed_dir / "rapt_evaluation"
        resnet_dir = seed_dir / "resnet_training" / "B2_resnet18_pretrained"
        rapt_checkpoint = seed_dir / "rapt_full" / "best.pt"
        resnet_checkpoint = resnet_dir / "best.pt"
        rapt_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        resnet_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        rapt_checkpoint.write_bytes(f"rapt-{seed}".encode())
        resnet_checkpoint.write_bytes(f"resnet-{seed}".encode())
        rapt_hash = hashlib.sha256(rapt_checkpoint.read_bytes()).hexdigest()
        resnet_hash = hashlib.sha256(resnet_checkpoint.read_bytes()).hexdigest()

        write_json(seed_dir / "COMPLETE.json", {
            "status": "PASS", "seed": seed, "test_access_protocol": "strict"
        })
        write_json(seed_dir / "commands.json", {"seed": seed})
        for stage in ("rapt_warmup", "rapt_full"):
            stage_dir = seed_dir / stage
            if stage == "rapt_warmup":
                (stage_dir / "best.pt").parent.mkdir(parents=True, exist_ok=True)
                (stage_dir / "best.pt").write_bytes(f"warmup-{seed}".encode())
            write_json(stage_dir / "preflight.json", {
                "test_access": "deferred_until_after_checkpoint_selection"
            })
            write_json(stage_dir / "selection_summary.json", {
                "selected_checkpoint_epoch": 1,
                "best_validation_macro_f1": 0.5,
                "test_access": "not_accessed",
            })
            write_csv(stage_dir / "epoch_metrics.csv", [{"epoch": 1, "validation_macro_f1": 0.5}])
        write_json(rapt_dir / "router_selection.json", {
            "selected": {"validation_macro_f1": 1.0},
            "candidates": [{"validation_accuracy": 1.0}],
        })
        write_json(resnet_dir / "preflight.json", {
            "dataset_sha256": DATASET_SHA256,
            "test_access": "deferred_until_after_checkpoint_selection",
        })
        write_csv(resnet_dir / "epoch_metrics.csv", [{"epoch": 1, "validation_macro_f1": 0.5}])
        (resnet_dir / "command.txt").write_text("synthetic\n", encoding="utf-8")

        rows_by_model: dict[str, dict[str, list[dict[str, object]]]] = {"rapt": {}, "resnet": {}}
        for split in ("validation", "test"):
            rapt = rapt_rows(split)
            for row in rapt:
                row["checkpoint_sha256"] = rapt_hash
                row["dataset_sha256"] = DATASET_SHA256
            resnet = resnet_rows(split)
            rows_by_model["rapt"][split] = rapt
            rows_by_model["resnet"][split] = resnet
            write_csv(rapt_dir / f"{split}_predictions.csv", rapt)
            write_csv(resnet_dir / f"predictions_{split}.csv", resnet)

            for model, file_model, checkpoint_hash, rows in (
                ("RAHAS-RAPT", "rapt", rapt_hash, rapt),
                ("ResNet-18", "resnet18", resnet_hash, resnet),
            ):
                copied = []
                for row in rows:
                    copy = dict(row)
                    copy.update({
                        "model": model, "model_seed": seed,
                        "checkpoint_sha256": checkpoint_hash,
                        "dataset_sha256": DATASET_SHA256,
                    })
                    copied.append(copy)
                write_csv(prediction_dir / f"{file_model}_seed{seed}_{split}.csv", copied)

        rapt_metrics = {
            split: metrics_from_predictions(rows_by_model["rapt"][split], True)
            for split in ("validation", "test")
        }
        resnet_metrics = {
            split: metrics_from_predictions(rows_by_model["resnet"][split], False)
            for split in ("validation", "test")
        }
        write_json(rapt_dir / "metrics.json", {
            "status": "PASS", "seed": seed, "dataset_sha256": DATASET_SHA256,
            "checkpoint_sha256": rapt_hash, **rapt_metrics,
            "hashes": {
                "class_map": "a" * 64, "train_manifest": "b" * 64,
                "validation_manifest": "c" * 64, "test_manifest": "d" * 64,
            },
        })
        write_json(resnet_dir / "metrics.json", {
            "status": "PASS", "seed": seed, "dataset_sha256": DATASET_SHA256,
            "checkpoint_sha256": resnet_hash, **resnet_metrics,
        })

        for relative in SEED_ARTIFACTS:
            path = seed_dir / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")
        if seed == 2026:
            write_csv(rapt_dir / "qualitative" / "qualitative_sheet_index.csv", [{"tile": 1}])

        paired: dict[str, object] = {"seed": seed}
        for metric in METRICS:
            rapt_value = rapt_metrics["test"][metric]
            resnet_value = resnet_metrics["test"][metric]
            paired[f"rapt_{metric}"] = rapt_value
            paired[f"resnet_{metric}"] = resnet_value
            paired[f"difference_{metric}"] = rapt_value - resnet_value
        paired_rows.append(paired)
        for model, values, checkpoint_hash in (
            ("RAHAS-RAPT", rapt_metrics, rapt_hash),
            ("ResNet-18", resnet_metrics, resnet_hash),
        ):
            row: dict[str, object] = {
                "model": model, "seed": seed, "checkpoint_sha256": checkpoint_hash,
                "dataset_sha256": DATASET_SHA256,
            }
            for split in ("validation", "test"):
                row.update({f"{split}_{name}": value for name, value in values[split].items()})
            repeated_rows.append(row)
            filename = model.lower().replace("-", "").replace(" ", "_")
            write_json(metric_dir / f"{filename}_seed{seed}.json", row)

    write_csv(root / "paired_comparison.csv", paired_rows)
    write_csv(root / "repeated_seed_summary.csv", repeated_rows)
    write_json(root / "statistical_summary.json", {"dataset_sha256": DATASET_SHA256})
    (root / "PHASE1_VALIDATION_REPORT.md").write_text("# Synthetic report\n", encoding="utf-8")
    write_csv(root / "tables" / "descriptive_statistics.csv", [{"model": "RAHAS-RAPT"}])


class Phase1VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        build_package(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_failed_with(self, fragment: str) -> None:
        result = verify_phase1(self.root)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(fragment in error for error in result["errors"]), result["errors"])

    def test_accepts_complete_internally_consistent_package(self):
        result = verify_phase1(self.root)
        self.assertEqual(result["status"], "PASS", result["errors"])

    def test_rejects_incomplete_seed(self):
        (self.root / "runs" / "seed_42" / "COMPLETE.json").unlink()
        self.assert_failed_with("seed_42")

    def test_rejects_prediction_path_misalignment(self):
        path = (
            self.root / "runs" / "seed_17" / "resnet_training"
            / "B2_resnet18_pretrained" / "predictions_test.csv"
        )
        rows = read_csv(path)
        rows[1]["sample_path"] = "images/wrong.png"
        write_csv(path, rows)
        self.assert_failed_with("test_model_alignment")

    def test_rejects_test_derived_selection_field(self):
        path = self.root / "runs" / "seed_123" / "rapt_evaluation" / "router_selection.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["selected"]["test_accuracy"] = 1.0
        write_json(path, value)
        self.assert_failed_with("test-derived selection keys")

    def test_rejects_inconsistent_prediction_hash(self):
        path = self.root / "seed_level_predictions" / "rapt_seed3407_test.csv"
        rows = read_csv(path)
        rows[0]["checkpoint_sha256"] = "f" * 64
        write_csv(path, rows)
        self.assert_failed_with("checkpoint_hash")

    def test_rejects_paired_difference_not_recomputed_value(self):
        path = self.root / "paired_comparison.csv"
        rows = read_csv(path)
        rows[0]["difference_accuracy"] = "0.123"
        write_csv(path, rows)
        self.assert_failed_with("difference")


if __name__ == "__main__":
    unittest.main()
