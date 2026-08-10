from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rahas_comparison_v1 import (
    CHARACTER_ROOT,
    DATASET_SHA256,
    EXPERIMENTS,
    SPLIT_DIR,
    assert_frozen_dataset,
    build_image_records_from_manifest,
    build_training_label_maps,
    evaluate,
    manifest_rows,
    parameter_count,
    sha256,
    write_csv,
)
from src.ocr.comparison_models import TorchvisionClassifier


EXPERIMENT = "B2_resnet18_pretrained"
EXPECTED_MODEL_CONFIG = {
    "architecture": "resnet18",
    "pretrained": True,
    "in_channels": 3,
    "num_classes": 372,
}
EVALUATION_ARTIFACTS = (
    "predictions_validation.csv",
    "predictions_test.csv",
    "metrics.json",
    "evaluation_command.txt",
)
LABEL_MAP_KEYS = ("idx_to_full_label", "idx_to_base_glyph", "idx_to_modifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the validation-selected Phase-1 ResNet after RAPT router selection."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-selection", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--characters", type=Path, default=CHARACTER_ROOT)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read required JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _key_paths(value: Any, prefix: str = "") -> list[str]:
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            paths.append(current)
            paths.extend(_key_paths(child, current))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_key_paths(child, f"{prefix}[{index}]"))
    return paths


def load_frozen_router(path: Path, expected_seed: int | None = None) -> dict[str, Any]:
    router = read_json(path)
    selected = router.get("selected")
    candidates = router.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("RAPT router selection is incomplete")
    required = {"max_transport_shots", "minimum_transport_margin", "validation_macro_f1"}
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"RAPT router selection is missing fields: {missing}")
    offenders = []
    for key_path in _key_paths(router):
        key = key_path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if "test" in key and key != "test_access":
            offenders.append(key_path)
    if offenders:
        raise ValueError(f"RAPT router selection contains test-derived fields: {offenders}")
    if expected_seed is not None:
        evaluation = read_json(path.parent / "metrics.json")
        expected = {
            "status": "PASS",
            "seed": expected_seed,
            "dataset_sha256": DATASET_SHA256,
        }
        for key, value in expected.items():
            if evaluation.get(key) != value:
                raise ValueError(
                    f"RAPT evaluation mismatch for {key}: "
                    f"expected {value!r}, found {evaluation.get(key)!r}"
                )
        if evaluation.get("router") != selected:
            raise ValueError("RAPT evaluation metrics do not match the frozen router selection")
    return router


def ensure_evaluation_targets_absent(output: Path) -> None:
    if not output.is_dir():
        raise FileNotFoundError(f"ResNet training output does not exist: {output}")
    existing = [str(output / name) for name in EVALUATION_ARTIFACTS if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Immutable ResNet evaluation artifacts already exist: {existing}")


def claim_test_access(marker: Path, seed: int) -> None:
    payload = {"status": "STARTED", "model": "ResNet-18", "seed": seed}
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"ResNet test access was already claimed at {marker}; "
            "restart this seed from a fresh immutable seed directory"
        ) from error


def class_map_sha256(maps: dict) -> str:
    payload = {key: maps[key] for key in LABEL_MAP_KEYS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_checkpoint(
    checkpoint: dict,
    selection: dict[str, Any],
    preflight: dict[str, Any],
    checkpoint_hash: str,
    args: argparse.Namespace,
) -> None:
    if checkpoint.get("experiment") != EXPERIMENT:
        raise ValueError(f"Expected checkpoint experiment {EXPERIMENT!r}")
    if checkpoint.get("spec") != EXPERIMENTS[EXPERIMENT]:
        raise ValueError("Checkpoint experiment specification does not match the frozen ResNet spec")
    if checkpoint.get("model_config") != EXPECTED_MODEL_CONFIG:
        raise ValueError(f"Checkpoint model configuration mismatch: {checkpoint.get('model_config')!r}")
    if checkpoint.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("Checkpoint dataset hash mismatch")
    training_args = checkpoint.get("args")
    if not isinstance(training_args, dict) or int(training_args.get("seed", -1)) != args.seed:
        raise ValueError("Checkpoint seed does not match requested evaluation seed")
    if training_args.get("experiment") != EXPERIMENT:
        raise ValueError("Checkpoint arguments do not identify the frozen ResNet experiment")
    if Path(training_args.get("split_dir", "")).resolve() != args.split_dir.resolve():
        raise ValueError("Checkpoint split directory does not match requested evaluation split")
    if Path(training_args.get("characters", "")).resolve() != args.characters.resolve():
        raise ValueError("Checkpoint character directory does not match requested evaluation data")
    if selection.get("test_access") != "not_accessed":
        raise ValueError("ResNet selection summary does not certify deferred test access")
    expected_selection = {
        "selected_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_score"]),
        "dataset_sha256": DATASET_SHA256,
        "checkpoint_sha256": checkpoint_hash,
        "seed": args.seed,
        "experiment": EXPERIMENT,
        "model": "resnet18",
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(
                f"ResNet selection summary mismatch for {key}: "
                f"expected {expected!r}, found {selection.get(key)!r}"
            )
    if preflight.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("ResNet preflight dataset hash mismatch")
    if preflight.get("seed") != args.seed or preflight.get("experiment") != EXPERIMENT:
        raise ValueError("ResNet preflight seed or experiment mismatch")
    if preflight.get("spec") != EXPERIMENTS[EXPERIMENT]:
        raise ValueError("ResNet preflight model specification mismatch")
    if preflight.get("test_access") != "deferred_until_after_checkpoint_selection":
        raise ValueError("ResNet preflight does not certify deferred test access")


def read_training_seconds(path: Path) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Epoch metrics are empty: {path}")
    return sum(float(row["training_seconds"]) for row in rows)


def add_provenance(rows: list[dict], seed: int, checkpoint_hash: str) -> None:
    for row in rows:
        row["model"] = "ResNet-18"
        row["model_seed"] = seed
        row["checkpoint_sha256"] = checkpoint_hash
        row["dataset_sha256"] = DATASET_SHA256


def add_one_shot_metric(metrics: dict[str, Any]) -> None:
    value = metrics.get("frequency_bins", {}).get("one_shot", {}).get("character_accuracy")
    metrics["one_shot_accuracy"] = float(value) if value is not None else 0.0


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.router_selection = args.router_selection.resolve()
    args.characters = args.characters.resolve()
    args.split_dir = args.split_dir.resolve()

    # This gate is deliberately first: ResNet evaluation cannot inspect any split
    # until the validation-selected RAPT router has been written.
    load_frozen_router(args.router_selection, args.seed)
    ensure_evaluation_targets_absent(args.output)
    expected_checkpoint = (args.output / "best.pt").resolve()
    if args.checkpoint != expected_checkpoint:
        raise ValueError(f"Evaluator must use the selected checkpoint at {expected_checkpoint}")

    selection = read_json(args.output / "selection_summary.json")
    preflight = read_json(args.output / "preflight.json")
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("ResNet checkpoint payload is not a dictionary")
    verify_checkpoint(checkpoint, selection, preflight, checkpoint_hash, args)

    frozen = assert_frozen_dataset(args.split_dir)
    maps = build_training_label_maps(args.characters)
    checkpoint_maps = checkpoint.get("label_maps")
    if not isinstance(checkpoint_maps, dict) or any(
        checkpoint_maps.get(key) != maps.get(key) for key in LABEL_MAP_KEYS
    ):
        raise ValueError("Checkpoint label maps do not match the frozen character map")

    project_root = Path.cwd().resolve()
    train_records = build_image_records_from_manifest(
        args.split_dir / "train_manifest.csv", maps, project_root, "train"
    )
    validation_records = build_image_records_from_manifest(
        args.split_dir / "validation_manifest.csv", maps, project_root, "validation"
    )
    train_rows, _ = manifest_rows(args.split_dir / "train_manifest.csv", project_root)
    _, validation_lookup = manifest_rows(
        args.split_dir / "validation_manifest.csv", project_root
    )
    original_train_counts = Counter(
        int(row["class_index"]) for row in train_rows if row["is_augmented"].lower() == "false"
    )

    training_args = argparse.Namespace(**checkpoint["args"])
    training_args.seed = args.seed
    training_args.workers = args.workers
    spec = EXPERIMENTS[EXPERIMENT]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchvisionClassifier("resnet18", pretrained=False, num_classes=372)
    model.pretrained = True
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    validation_metrics, validation_predictions = evaluate(
        model,
        validation_records,
        None,
        spec,
        training_args,
        maps,
        validation_lookup,
        original_train_counts,
        device,
    )

    # A crash after this durable claim is fail-closed: restart the seed instead
    # of evaluating the test set a second time.
    claim_test_access(args.output / "TEST_ACCESS_STARTED.json", args.seed)
    # The test manifest is opened exactly once, after both model checkpoints and
    # the RAPT router have been frozen.
    test_records = build_image_records_from_manifest(
        args.split_dir / "test_manifest.csv", maps, project_root, "test"
    )
    _, test_lookup = manifest_rows(args.split_dir / "test_manifest.csv", project_root)
    test_metrics, test_predictions = evaluate(
        model,
        test_records,
        None,
        spec,
        training_args,
        maps,
        test_lookup,
        original_train_counts,
        device,
    )
    add_one_shot_metric(validation_metrics)
    add_one_shot_metric(test_metrics)
    add_provenance(validation_predictions, args.seed, checkpoint_hash)
    add_provenance(test_predictions, args.seed, checkpoint_hash)

    result = {
        "status": "PASS",
        "protocol": "validation-only checkpoint selection; test opened after RAPT router selection",
        "experiment_id": EXPERIMENT,
        "model": "resnet18",
        "input_representation": spec["representation"],
        "description": spec["description"],
        "dataset_sha256": DATASET_SHA256,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": validation_metrics,
        "test": test_metrics,
        "training_seconds": read_training_seconds(args.output / "epoch_metrics.csv"),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(args.checkpoint),
        "router_selection_sha256": sha256(args.router_selection),
        "hashes": {
            "class_map": class_map_sha256(maps),
            "train_manifest": frozen["manifest_sha256"]["train_manifest.csv"],
            "validation_manifest": frozen["manifest_sha256"]["validation_manifest.csv"],
            "test_manifest": frozen["manifest_sha256"]["test_manifest.csv"],
        },
        "protocol_deviations": [],
    }
    write_csv(args.output / "predictions_validation.csv", validation_predictions)
    write_csv(args.output / "predictions_test.csv", test_predictions)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "evaluation_command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    print(
        f"RESNET PHASE1 seed={args.seed} val_macro_f1={validation_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
