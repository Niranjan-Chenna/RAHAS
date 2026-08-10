from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_rahas_resnet_phase1_v1 as resnet_evaluator
import evaluate_rahas_rapt_phase1_v1 as rapt_evaluator
import run_rahas_comparison_v1 as comparison_runner
import run_rahas_rapt_phase1_seed_v1 as seed_runner


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class Phase1SchedulingTests(unittest.TestCase):
    def test_comparison_parser_supports_skip_test(self):
        argv = [
            "run_rahas_comparison_v1.py",
            "--experiment",
            "B2_resnet18_pretrained",
            "--skip-test",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = comparison_runner.parse_args()
        self.assertTrue(args.skip_test)

    def test_comparison_skip_test_does_not_open_test_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            split_dir = Path(temporary)
            train = split_dir / "train_manifest.csv"
            train.write_text("sample_path\n", encoding="utf-8")
            train_hash = hashlib.sha256(train.read_bytes()).hexdigest()
            write_json(
                split_dir / "split_summary.json",
                {
                    "dataset_sha256": comparison_runner.DATASET_SHA256,
                    "manifest_sha256": {
                        "train_manifest.csv": train_hash,
                        "test_manifest.csv": "f" * 64,
                    },
                },
            )
            write_json(
                split_dir / "leakage_audit.json",
                {"status": "PASS", "checks": {"source_disjoint": True}},
            )

            frozen = comparison_runner.assert_frozen_dataset(
                split_dir, include_test=False
            )
            self.assertEqual(
                frozen["manifest_sha256"]["test_manifest.csv"], "f" * 64
            )
            with self.assertRaises(FileNotFoundError):
                comparison_runner.assert_frozen_dataset(split_dir)

    def test_seed_runner_freezes_then_evaluates_each_test_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "phase1"
            args = argparse.Namespace(
                seed=17,
                output_root=output_root,
                workers=0,
                resume=False,
            )
            stages: list[tuple[str, list[str]]] = []

            def fake_stage(name: str, command: list[str], log_path: Path) -> None:
                stages.append((name, command))
                script = Path(command[1]).name
                if script == "run_rahas_comparison_v1.py":
                    root = Path(command[command.index("--output-root") + 1])
                    write_json(
                        root / "B2_resnet18_pretrained" / "selection_summary.json",
                        {"test_access": "not_accessed"},
                    )
                elif script == "train_rahas_rapt_v1.py":
                    output = Path(command[command.index("--output") + 1])
                    write_json(output / "selection_summary.json", {"test_access": "not_accessed"})
                elif script == "evaluate_rahas_rapt_phase1_v1.py":
                    output = Path(command[command.index("--output") + 1])
                    write_json(output / "router_selection.json", {"selected": {}, "candidates": [{}]})
                    write_json(output / "metrics.json", {"status": "PASS"})
                elif script == "evaluate_rahas_resnet_phase1_v1.py":
                    output = Path(command[command.index("--output") + 1])
                    self.assertTrue(Path(command[command.index("--router-selection") + 1]).is_file())
                    write_json(output / "metrics.json", {"status": "PASS"})
                else:
                    self.fail(f"Unexpected stage command: {command}")

            with (
                mock.patch.object(seed_runner, "parse_args", return_value=args),
                mock.patch.object(seed_runner, "run_stage", side_effect=fake_stage),
            ):
                seed_runner.main()

            scripts = [Path(command[1]).name for _, command in stages]
            self.assertEqual(
                scripts,
                [
                    "run_rahas_comparison_v1.py",
                    "train_rahas_rapt_v1.py",
                    "train_rahas_rapt_v1.py",
                    "evaluate_rahas_rapt_phase1_v1.py",
                    "evaluate_rahas_resnet_phase1_v1.py",
                ],
            )
            resnet_training = stages[0][1]
            self.assertIn("--skip-test", resnet_training)
            rapt_evaluation = stages[3][1]
            self.assertNotIn("--resnet-validation", rapt_evaluation)
            self.assertNotIn("--resnet-test", rapt_evaluation)
            self.assertEqual(scripts.count("evaluate_rahas_rapt_phase1_v1.py"), 1)
            self.assertEqual(scripts.count("evaluate_rahas_resnet_phase1_v1.py"), 1)

    def test_resnet_evaluator_requires_validation_only_router(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "router_selection.json"
            router = {
                    "selected": {
                        "max_transport_shots": 1,
                        "minimum_transport_margin": 0.1,
                        "validation_macro_f1": 0.5,
                    },
                    "candidates": [{"validation_macro_f1": 0.5}],
                }
            write_json(path, router)
            write_json(
                path.parent / "metrics.json",
                {
                    "status": "PASS",
                    "seed": 17,
                    "dataset_sha256": resnet_evaluator.DATASET_SHA256,
                    "router": router["selected"],
                },
            )
            value = resnet_evaluator.load_frozen_router(path, expected_seed=17)
            self.assertEqual(value["selected"]["max_transport_shots"], 1)
            value["selected"]["test_accuracy"] = 1.0
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "test-derived"):
                resnet_evaluator.load_frozen_router(path)

    def test_resnet_evaluator_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            resnet_evaluator.ensure_evaluation_targets_absent(output)
            (output / "predictions_test.csv").write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                resnet_evaluator.ensure_evaluation_targets_absent(output)

    def test_test_access_claims_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for claim, marker in (
                (rapt_evaluator.claim_test_access, root / "rapt_started.json"),
                (resnet_evaluator.claim_test_access, root / "resnet_started.json"),
            ):
                claim(marker, 17)
                self.assertTrue(marker.is_file())
                with self.assertRaisesRegex(RuntimeError, "already claimed"):
                    claim(marker, 17)


if __name__ == "__main__":
    unittest.main()
