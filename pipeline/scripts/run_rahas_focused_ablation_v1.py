from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from summarize_rahas_rapt_phase1_v1 import (  # noqa: E402
    DATASET_SHA256,
    SEEDS,
    read_csv,
    require_exact_model_alignment,
    sha256,
    validate_prediction_rows,
)


DEFAULT_OUTPUT_ROOT = Path("pipeline/experiments/rahas_rapt_focused_ablation_v1")
DEFAULT_PHASE1_ROOT = Path("pipeline/experiments/rahas_rapt_validation_v1")
TRAIN_ARTIFACTS = ("best.pt", "selection_summary.json", "epoch_metrics.csv")
EVALUATION_ARTIFACTS = ("metrics.json", "validation_predictions.csv", "test_predictions.csv")
MODE_FLAG_CANDIDATES = (
    "--routing-mode",
    "--router-mode",
    "--routing-policy",
    "--evaluation-mode",
    "--inference-mode",
    "--decision-mode",
    "--prediction-mode",
    "--mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict five-seed focused RAHAS-RAPT ablation study."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume checkpointed training and skip only hash-verified completed stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify Phase 1 inputs and print commands without creating focused outputs.",
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _normalized(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, found {actual!r}")


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required stage artifact is missing: {path}")
        result[_normalized(path)] = sha256(path)
    return result


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(f"immutable file already exists with different content: {path}")


def _validate_metrics_metadata(
    path: Path, *, seed: int, checkpoint_hash: str, model: str
) -> dict[str, Any]:
    metrics = _json(path)
    if "status" in metrics:
        _require_equal(metrics.get("status"), "PASS", f"{path} status")
    _require_equal(metrics.get("seed"), seed, f"{path} seed")
    _require_equal(metrics.get("dataset_sha256"), DATASET_SHA256, f"{path} dataset hash")
    _require_equal(
        metrics.get("checkpoint_sha256"), checkpoint_hash, f"{path} checkpoint hash"
    )
    if not isinstance(metrics.get("test"), dict) and "test_accuracy" not in metrics:
        raise ValueError(f"{path}: missing test metrics")
    if model not in {"RAHAS-RAPT", "ResNet-18"}:
        raise ValueError(f"unsupported model metadata check: {model}")
    return metrics


def verify_phase1_seed(phase1_root: Path, seed: int) -> dict[str, Path | str]:
    seed_dir = phase1_root.resolve() / "runs" / f"seed_{seed}"
    complete = _json(seed_dir / "COMPLETE.json")
    _require_equal(complete.get("status"), "PASS", f"seed {seed} completion status")
    _require_equal(complete.get("seed"), seed, f"seed {seed} completion seed")
    _require_equal(
        complete.get("test_access_protocol"), "strict", f"seed {seed} test protocol"
    )

    rapt_checkpoint = seed_dir / "rapt_full" / "best.pt"
    resnet_dir = seed_dir / "resnet_training" / "B2_resnet18_pretrained"
    resnet_checkpoint = resnet_dir / "best.pt"
    rapt_hash = sha256(rapt_checkpoint)
    resnet_hash = sha256(resnet_checkpoint)
    rapt_metrics = phase1_root.resolve() / "seed_level_metrics" / f"rahasrapt_seed{seed}.json"
    resnet_metrics = phase1_root.resolve() / "seed_level_metrics" / f"resnet18_seed{seed}.json"
    _validate_metrics_metadata(
        rapt_metrics, seed=seed, checkpoint_hash=rapt_hash, model="RAHAS-RAPT"
    )
    _validate_metrics_metadata(
        resnet_metrics, seed=seed, checkpoint_hash=resnet_hash, model="ResNet-18"
    )

    rapt_predictions = phase1_root.resolve() / "seed_level_predictions" / f"rapt_seed{seed}_test.csv"
    resnet_predictions = phase1_root.resolve() / "seed_level_predictions" / f"resnet18_seed{seed}_test.csv"
    rapt_rows = read_csv(rapt_predictions)
    resnet_rows = read_csv(resnet_predictions)
    validate_prediction_rows(
        rapt_rows,
        True,
        context=f"Phase 1 seed {seed} full RAPT test",
        seed=seed,
        split="test",
        checkpoint_hash=rapt_hash,
    )
    validate_prediction_rows(
        resnet_rows,
        False,
        context=f"Phase 1 seed {seed} ResNet test",
        seed=seed,
        split="test",
        checkpoint_hash=resnet_hash,
    )
    require_exact_model_alignment(
        rapt_rows, resnet_rows, context=f"Phase 1 seed {seed} test"
    )
    return {
        "seed_dir": seed_dir,
        "rapt_checkpoint": rapt_checkpoint,
        "rapt_checkpoint_sha256": rapt_hash,
        "rapt_metrics": rapt_metrics,
        "rapt_predictions": rapt_predictions,
        "resnet_checkpoint": resnet_checkpoint,
        "resnet_checkpoint_sha256": resnet_hash,
        "resnet_metrics": resnet_metrics,
        "resnet_predictions": resnet_predictions,
    }


def evaluator_mode_flag(evaluator: Path) -> str:
    source = evaluator.read_text(encoding="utf-8")
    for flag in MODE_FLAG_CANDIDATES:
        if repr(flag) in source or f'"{flag}"' in source:
            return flag
    # This is the shared evaluator interface expected by this focused study.
    # Keeping the default here makes command generation independently testable
    # while main() still fails early until the interface is present.
    return "--routing-mode"


def ensure_focused_interfaces(trainer: Path, evaluator: Path) -> str:
    trainer_source = trainer.read_text(encoding="utf-8")
    if "--disable-prototype-completion" not in trainer_source:
        raise RuntimeError(
            f"{trainer} does not yet expose --disable-prototype-completion"
        )
    mode_flag = evaluator_mode_flag(evaluator)
    evaluator_source = evaluator.read_text(encoding="utf-8")
    if mode_flag not in evaluator_source:
        raise RuntimeError(f"{evaluator} does not yet expose a focused routing-mode flag")
    return mode_flag


def _training_base(python: str, trainer: Path, output: Path, seed: int, workers: int) -> list[str]:
    return [
        python,
        str(trainer),
        "--output",
        str(output),
        "--episodes-per-epoch",
        "30",
        "--n-way",
        "16",
        "--support-shots",
        "1,2,4",
        "--q-query",
        "1",
        "--token-dim",
        "128",
        "--embedding-dim",
        "256",
        "--prototype-per-class",
        "5",
        "--lr",
        "0.0002",
        "--backbone-lr-scale",
        "0.2",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--disable-prototype-completion",
        "--skip-test",
    ]


def build_seed_commands(
    seed: int,
    seed_dir: Path,
    phase1: dict[str, Path | str],
    *,
    python: str,
    workers: int,
    mode_flag: str = "--routing-mode",
) -> dict[str, list[str]]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, found {seed}")
    evaluator = ROOT / "pipeline/scripts/evaluate_rahas_rapt_phase1_v1.py"
    trainer = ROOT / "pipeline/scripts/train_rahas_rapt_v1.py"
    full_checkpoint = Path(phase1["rapt_checkpoint"])
    resnet_checkpoint = Path(phase1["resnet_checkpoint"])
    warmup = seed_dir / "no_prototype_completion_warmup"
    full = seed_dir / "no_prototype_completion_full"

    def evaluation(output: Path, checkpoint: Path, mode: str) -> list[str]:
        return [
            python,
            str(evaluator),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--workers",
            str(workers),
            mode_flag,
            mode,
            "--experiment-id",
            f"focused_ablation_{mode}_seed_{seed}",
        ]

    warmup_command = [
        *_training_base(python, trainer, warmup, seed, workers),
        "--epochs",
        "12",
    ]
    full_command = [
        *_training_base(python, trainer, full, seed, workers),
        "--epochs",
        "6",
        "--classification-batches-per-epoch",
        "67",
        "--classification-batch-size",
        "64",
        "--warm-start-rapt",
        str(warmup / "best.pt"),
        "--resnet-classifier-init",
        str(resnet_checkpoint),
    ]
    return {
        "no_adaptive_routing": evaluation(
            seed_dir / "no_adaptive_routing", full_checkpoint, "equal_fusion"
        ),
        "no_prototype_completion_warmup": warmup_command,
        "no_prototype_completion_full": full_command,
        "no_prototype_completion_evaluation": evaluation(
            seed_dir / "no_prototype_completion_evaluation", full / "best.pt", "routed"
        ),
        "no_low_shot_specialist": evaluation(
            seed_dir / "no_low_shot_specialist", full_checkpoint, "direct_only"
        ),
    }


def _contains_test_selection(value: Any, prefix: str = "") -> list[str]:
    offenders: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).lower()
            if "test" in normalized and normalized != "test_access":
                offenders.append(current)
            offenders.extend(_contains_test_selection(child, current))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            offenders.extend(_contains_test_selection(child, f"{prefix}[{index}]"))
    return offenders


def validate_training_outputs(output: Path) -> None:
    selection = _json(output / "selection_summary.json")
    offenders = _contains_test_selection(selection)
    if offenders:
        raise RuntimeError(f"training selection contains test-derived fields: {offenders}")
    with (output / "epoch_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    test_columns = [name for name in fieldnames if "test" in name.lower()]
    if test_columns:
        raise RuntimeError(f"training epoch metrics contain test columns: {test_columns}")
    test_markers = list(output.parent.glob(f"{output.name}*TEST_ACCESS_STARTED.json"))
    if test_markers:
        raise RuntimeError(f"training stage accessed test material: {test_markers}")


def _resume_training_command(command: list[str], checkpoint: Path) -> list[str]:
    resumed: list[str] = []
    skip = False
    for value in command:
        if skip:
            skip = False
            continue
        if value in {"--warm-start-rapt", "--resnet-classifier-init"}:
            skip = True
            continue
        resumed.append(value)
    return [*resumed, "--resume", str(checkpoint)]


def run_command(label: str, command: list[str], log_path: Path, *, append: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {label} =====", flush=True)
    rendered = subprocess.list2cmdline(command)
    print(rendered, flush=True)
    mode = "a" if append else "x"
    with log_path.open(mode, encoding="utf-8", newline="\n") as log:
        log.write(rendered + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"stage {label!r} failed with exit code {code}; see {log_path}")


def _verify_marker(
    marker: Path, *, seed: int, stage: str, command: list[str], inputs: dict[str, str]
) -> None:
    value = _json(marker)
    expected = {
        "schema_version": 1,
        "status": "PASS",
        "seed": seed,
        "stage": stage,
        "command": command,
        "inputs": inputs,
    }
    for key, required in expected.items():
        _require_equal(value.get(key), required, f"{marker} {key}")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{marker}: missing artifact hashes")
    for name, expected_hash in artifacts.items():
        path = Path(name)
        _require_equal(sha256(path), expected_hash, f"{marker} artifact {path}")


def _publish_marker(
    marker: Path,
    *,
    seed: int,
    stage: str,
    command: list[str],
    executed_command: list[str],
    inputs: dict[str, str],
    artifacts: Iterable[Path],
) -> None:
    _write_new_json(
        marker,
        {
            "schema_version": 1,
            "status": "PASS",
            "seed": seed,
            "stage": stage,
            "command": command,
            "executed_command": executed_command,
            "inputs": inputs,
            "artifacts": _artifact_hashes(artifacts),
        },
    )


def _run_stage(
    *,
    seed: int,
    stage: str,
    command: list[str],
    output: Path,
    artifacts: tuple[str, ...],
    inputs: dict[str, str],
    marker: Path,
    log: Path,
    resume: bool,
    training: bool,
) -> None:
    if marker.exists():
        if not resume:
            raise FileExistsError(f"immutable stage marker already exists: {marker}")
        _verify_marker(marker, seed=seed, stage=stage, command=command, inputs=inputs)
        print(f"SKIP hash-verified stage: seed={seed} {stage}", flush=True)
        return

    executed = command
    append_log = False
    if output.exists() and any(output.iterdir()):
        latest = output / "latest.pt"
        if resume and training and latest.is_file():
            executed = _resume_training_command(command, latest)
            append_log = log.exists()
        else:
            raise RuntimeError(f"cannot resume incomplete immutable stage directory: {output}")
    run_command(f"seed {seed} {stage}", executed, log, append=append_log)
    if training:
        validate_training_outputs(output)
    artifact_paths = tuple(output / name for name in artifacts)
    _publish_marker(
        marker,
        seed=seed,
        stage=stage,
        command=command,
        executed_command=executed,
        inputs=inputs,
        artifacts=artifact_paths,
    )


def _phase1_marker_artifacts(phase1: dict[str, Path | str]) -> tuple[Path, ...]:
    return tuple(
        Path(phase1[key])
        for key in (
            "rapt_checkpoint",
            "rapt_metrics",
            "rapt_predictions",
            "resnet_checkpoint",
            "resnet_metrics",
            "resnet_predictions",
        )
    )


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    output_root = args.output_root.resolve()
    phase1_root = args.phase1_root.resolve()
    trainer = ROOT / "pipeline/scripts/train_rahas_rapt_v1.py"
    evaluator = ROOT / "pipeline/scripts/evaluate_rahas_rapt_phase1_v1.py"
    mode_flag = ensure_focused_interfaces(trainer, evaluator)

    verified = {seed: verify_phase1_seed(phase1_root, seed) for seed in SEEDS}
    commands = {
        str(seed): build_seed_commands(
            seed,
            output_root / "runs" / f"seed_{seed}",
            verified[seed],
            python=sys.executable,
            workers=args.workers,
            mode_flag=mode_flag,
        )
        for seed in SEEDS
    }
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"immutable focused output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=args.resume)
    _write_new_json(output_root / "commands.json", commands)

    for seed in SEEDS:
        seed_dir = output_root / "runs" / f"seed_{seed}"
        stages_dir = seed_dir / "stages"
        logs_dir = seed_dir / "logs"
        seed_dir.mkdir(parents=True, exist_ok=args.resume)
        phase1 = verified[seed]
        reuse_inputs = {
            "phase1_root": _normalized(phase1_root),
            "dataset_sha256": DATASET_SHA256,
            "rapt_checkpoint_sha256": str(phase1["rapt_checkpoint_sha256"]),
            "resnet_checkpoint_sha256": str(phase1["resnet_checkpoint_sha256"]),
        }
        reuse_marker = stages_dir / "phase1_reuse.complete.json"
        reuse_command: list[str] = []
        if reuse_marker.exists():
            if not args.resume:
                raise FileExistsError(reuse_marker)
            _verify_marker(
                reuse_marker,
                seed=seed,
                stage="phase1_reuse",
                command=reuse_command,
                inputs=reuse_inputs,
            )
        else:
            _publish_marker(
                reuse_marker,
                seed=seed,
                stage="phase1_reuse",
                command=reuse_command,
                executed_command=reuse_command,
                inputs=reuse_inputs,
                artifacts=_phase1_marker_artifacts(phase1),
            )

        seed_commands = commands[str(seed)]
        rapt_hash = str(phase1["rapt_checkpoint_sha256"])
        resnet_hash = str(phase1["resnet_checkpoint_sha256"])
        stage_specs = (
            (
                "no_adaptive_routing",
                False,
                EVALUATION_ARTIFACTS,
                {"checkpoint_sha256": rapt_hash, "evaluation_mode": "equal_fusion"},
            ),
            (
                "no_prototype_completion_warmup",
                True,
                TRAIN_ARTIFACTS,
                {"dataset_sha256": DATASET_SHA256, "prototype_completion": "disabled"},
            ),
            (
                "no_prototype_completion_full",
                True,
                TRAIN_ARTIFACTS,
                {
                    "dataset_sha256": DATASET_SHA256,
                    "prototype_completion": "disabled",
                    "resnet_checkpoint_sha256": resnet_hash,
                },
            ),
            (
                "no_prototype_completion_evaluation",
                False,
                EVALUATION_ARTIFACTS,
                {"evaluation_mode": "routed"},
            ),
            (
                "no_low_shot_specialist",
                False,
                EVALUATION_ARTIFACTS,
                {"checkpoint_sha256": rapt_hash, "evaluation_mode": "direct_only"},
            ),
        )
        for stage, training, artifacts, inputs in stage_specs:
            if stage == "no_prototype_completion_full":
                inputs = {
                    **inputs,
                    "warmup_checkpoint_sha256": sha256(
                        seed_dir / "no_prototype_completion_warmup" / "best.pt"
                    ),
                }
            elif stage == "no_prototype_completion_evaluation":
                inputs = {
                    **inputs,
                    "checkpoint_sha256": sha256(
                        seed_dir / "no_prototype_completion_full" / "best.pt"
                    ),
                }
            output = seed_dir / stage
            _run_stage(
                seed=seed,
                stage=stage,
                command=seed_commands[stage],
                output=output,
                artifacts=artifacts,
                inputs=inputs,
                marker=stages_dir / f"{stage}.complete.json",
                log=logs_dir / f"{stage}.log",
                resume=args.resume,
                training=training,
            )
        _write_new_json(
            seed_dir / "COMPLETE.json",
            {
                "status": "PASS",
                "seed": seed,
                "dataset_sha256": DATASET_SHA256,
                "test_access_protocol": "strict",
                "stages": ["phase1_reuse", *(item[0] for item in stage_specs)],
            },
        )
        print(f"SEED {seed} COMPLETE: {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
