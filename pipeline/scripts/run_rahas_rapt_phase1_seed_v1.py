from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one strict repeated-seed RAPT versus ResNet comparison.")
    parser.add_argument("--seed", type=int, required=True, choices=(2026, 17, 42, 123, 3407))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pipeline/experiments/rahas_rapt_validation_v1"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Resume only from fully completed immutable stages.")
    return parser.parse_args()


def run_stage(name: str, command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {name} =====", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(command) + "\n\n")
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
        raise RuntimeError(f"Stage {name!r} failed with exit code {code}; see {log_path}")


def checkpoint_resume_command(command: list[str], checkpoint: Path) -> list[str]:
    resumed = []
    skip_next = False
    for value in command:
        if skip_next:
            skip_next = False
            continue
        if value in {"--warm-start-rapt", "--resnet-classifier-init"}:
            skip_next = True
            continue
        resumed.append(value)
    return [*resumed, "--resume", str(checkpoint)]


def main() -> None:
    args = parse_args()
    seed_dir = (args.output_root / "runs" / f"seed_{args.seed}").resolve()
    if seed_dir.exists() and not args.resume:
        raise FileExistsError(f"Immutable seed directory already exists: {seed_dir}")
    seed_dir.mkdir(parents=True, exist_ok=args.resume)
    logs = seed_dir / "logs"
    python = sys.executable

    resnet_root = seed_dir / "resnet_training"
    resnet_dir = resnet_root / "B2_resnet18_pretrained"
    warmup_dir = seed_dir / "rapt_warmup"
    rapt_dir = seed_dir / "rapt_full"
    evaluation_dir = seed_dir / "rapt_evaluation"

    commands = {
        "resnet": [
            python,
            str(ROOT / "pipeline/scripts/run_rahas_comparison_v1.py"),
            "--experiment",
            "B2_resnet18_pretrained",
            "--output-root",
            str(resnet_root),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
            "--skip-test",
        ],
        "rapt_warmup": [
            python,
            str(ROOT / "pipeline/scripts/train_rahas_rapt_v1.py"),
            "--output",
            str(warmup_dir),
            "--epochs",
            "12",
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
            str(args.seed),
            "--workers",
            str(args.workers),
            "--skip-test",
        ],
        "rapt_full": [
            python,
            str(ROOT / "pipeline/scripts/train_rahas_rapt_v1.py"),
            "--output",
            str(rapt_dir),
            "--epochs",
            "6",
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
            "--classification-batches-per-epoch",
            "67",
            "--classification-batch-size",
            "64",
            "--lr",
            "0.0002",
            "--backbone-lr-scale",
            "0.2",
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
            "--warm-start-rapt",
            str(warmup_dir / "best.pt"),
            "--resnet-classifier-init",
            str(resnet_dir / "best.pt"),
            "--skip-test",
        ],
        "rapt_evaluation": [
            python,
            str(ROOT / "pipeline/scripts/evaluate_rahas_rapt_phase1_v1.py"),
            "--checkpoint",
            str(rapt_dir / "best.pt"),
            "--output",
            str(evaluation_dir),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
        ],
        "resnet_evaluation": [
            python,
            str(ROOT / "pipeline/scripts/evaluate_rahas_resnet_phase1_v1.py"),
            "--checkpoint",
            str(resnet_dir / "best.pt"),
            "--output",
            str(resnet_dir),
            "--router-selection",
            str(evaluation_dir / "router_selection.json"),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
        ],
    }
    if args.seed == 2026:
        commands["rapt_evaluation"].append("--qualitative")
    (seed_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")

    stages = (
        ("pretrained ResNet-18 selection", "resnet", logs / "resnet.log", resnet_dir / "selection_summary.json"),
        ("RAPT transport warmup", "rapt_warmup", logs / "rapt_warmup.log", warmup_dir / "selection_summary.json"),
        ("full RAPT", "rapt_full", logs / "rapt_full.log", rapt_dir / "selection_summary.json"),
        ("validation-selected RAPT evaluation", "rapt_evaluation", logs / "rapt_evaluation.log", evaluation_dir / "metrics.json"),
        ("validation-selected ResNet evaluation", "resnet_evaluation", logs / "resnet_evaluation.log", resnet_dir / "metrics.json"),
    )
    for label, key, log_path, marker in stages:
        if args.resume and marker.exists():
            print(f"SKIP completed stage: {label} ({marker})", flush=True)
            continue
        command = commands[key]
        latest = marker.parent / "latest.pt"
        if args.resume and latest.exists() and key in {"rapt_warmup", "rapt_full"}:
            command = checkpoint_resume_command(command, latest)
            print(f"RESUME checkpointed stage: {label} ({latest})", flush=True)
        elif (
            args.resume
            and marker.parent.exists()
            and any(marker.parent.iterdir())
            and key in {"resnet", "rapt_warmup", "rapt_full", "rapt_evaluation"}
        ):
            raise RuntimeError(f"Cannot resume incomplete immutable stage {label}: {marker.parent}")
        run_stage(label, command, log_path)
    (seed_dir / "COMPLETE.json").write_text(
        json.dumps({"status": "PASS", "seed": args.seed, "test_access_protocol": "strict"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SEED {args.seed} COMPLETE: {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
