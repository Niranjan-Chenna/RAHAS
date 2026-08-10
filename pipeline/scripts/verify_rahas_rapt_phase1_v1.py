from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SEEDS = (2026, 17, 42, 123, 3407)
DATASET_SHA256 = "4d241e39f754b8cb4271eb94194eb07a706d50ccf61cf966063e87f91b0a8d7b"
METRICS = ("accuracy", "macro_f1", "top3", "top5", "one_shot_accuracy", "base_accuracy", "modifier_accuracy")
EVALUATION_METRICS = ("accuracy", "macro_f1", "top3", "top5", "base_accuracy", "modifier_accuracy")
TOLERANCE = 1e-9
SEED_ARTIFACTS = (
    "COMPLETE.json", "commands.json",
    "resnet_training/B2_resnet18_pretrained/best.pt",
    "resnet_training/B2_resnet18_pretrained/epoch_metrics.csv",
    "resnet_training/B2_resnet18_pretrained/metrics.json",
    "resnet_training/B2_resnet18_pretrained/preflight.json",
    "resnet_training/B2_resnet18_pretrained/predictions_validation.csv",
    "resnet_training/B2_resnet18_pretrained/predictions_test.csv",
    "resnet_training/B2_resnet18_pretrained/command.txt",
    "rapt_warmup/best.pt", "rapt_warmup/epoch_metrics.csv",
    "rapt_warmup/preflight.json", "rapt_warmup/selection_summary.json",
    "rapt_full/best.pt", "rapt_full/epoch_metrics.csv",
    "rapt_full/preflight.json", "rapt_full/selection_summary.json",
    "rapt_evaluation/metrics.json", "rapt_evaluation/router_selection.json",
    "rapt_evaluation/validation_predictions.csv", "rapt_evaluation/test_predictions.csv",
    "rapt_evaluation/validation_per_class_metrics.csv", "rapt_evaluation/test_per_class_metrics.csv",
    "rapt_evaluation/frequency_metrics.csv", "rapt_evaluation/inscription_metrics.csv",
    "rapt_evaluation/route_metrics.csv", "rapt_evaluation/confusion_matrix.csv",
    "rapt_evaluation/confusion_matrix_normalized.csv",
)
SUMMARY_ARTIFACTS = (
    "repeated_seed_summary.csv", "paired_comparison.csv", "statistical_summary.json",
    "PHASE1_VALIDATION_REPORT.md", "tables/descriptive_statistics.csv",
)


class Audit:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.errors: list[str] = []

    def check(self, name: str, condition: bool, detail: str) -> bool:
        passed = bool(condition)
        self.checks[name] = self.checks.get(name, True) and passed
        if not passed:
            self.errors.append(f"{name}: {detail}")
        return passed

    def json(self, path: Path, name: str) -> Any | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self.check(name, False, f"cannot read {path}: {error}")
            return None

    def csv(self, path: Path, name: str) -> list[dict[str, str]] | None:
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as error:
            self.check(name, False, f"cannot read {path}: {error}")
            return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> str:
    return value.replace("\\", "/")


def _hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _close(left: object, right: object) -> bool:
    try:
        a, b = float(left), float(right)
        return math.isfinite(a) and math.isfinite(b) and math.isclose(
            a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE
        )
    except (TypeError, ValueError):
        return False


def metrics_from_predictions(rows: list[dict[str, str]], rapt: bool) -> dict[str, float]:
    """Recompute summary metrics without importing evaluator or summarizer code."""
    if not rows:
        raise ValueError("prediction CSV has no rows")
    target_key = "true_class_index" if rapt else "class_index"
    label_key = "true_full_label" if rapt else "class_label"
    base_true, base_pred = (
        ("true_base_label", "predicted_base_from_full")
        if rapt
        else ("true_base", "predicted_base")
    )
    mod_true, mod_pred = (
        ("true_modifier_label", "predicted_modifier_from_full")
        if rapt
        else ("true_modifier", "predicted_modifier")
    )
    targets = [int(row[target_key]) for row in rows]
    predictions = [int(row["predicted_class_index"]) for row in rows]
    f1s = []
    for label in sorted(set(targets)):
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    def top_hit(row: dict[str, str], limit: int) -> bool:
        legacy = row.get("top5_labels", "").split("|")
        ranked = [
            row.get(f"top_{rank}_label") or (legacy[rank - 1] if len(legacy) >= rank else "")
            for rank in range(1, limit + 1)
        ]
        return row[label_key] in ranked

    one_shot = [
        t == p for row, t, p in zip(rows, targets, predictions)
        if row["frequency_bin"] == "one_shot"
    ]
    total = len(rows)
    return {
        "accuracy": sum(t == p for t, p in zip(targets, predictions)) / total,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "top3": sum(top_hit(row, 3) for row in rows) / total,
        "top5": sum(top_hit(row, 5) for row in rows) / total,
        "one_shot_accuracy": sum(one_shot) / len(one_shot) if one_shot else 0.0,
        "base_accuracy": sum(row[base_true] == row[base_pred] for row in rows) / total,
        "modifier_accuracy": sum(row[mod_true] == row[mod_pred] for row in rows) / total,
    }


def _key_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            yield current
            yield from _key_paths(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _key_paths(child, f"{prefix}[{index}]")


def _selection(audit: Audit, path: Path, name: str, csv_file: bool = False) -> None:
    value = audit.csv(path, name) if csv_file else audit.json(path, name)
    if value is None:
        return
    offenders = []
    for key_path in _key_paths(value):
        key = key_path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if "test" in key and key != "test_access":
            offenders.append(key_path)
    audit.check(f"{name}.no_test_fields", not offenders, f"test-derived selection keys: {offenders}")


def _required_metrics(audit: Audit, value: Any, name: str) -> None:
    missing = [metric for metric in EVALUATION_METRICS if not isinstance(value, dict) or metric not in value]
    audit.check(name, not missing, f"missing metrics: {missing}")


def _row_hashes(
    audit: Audit, rows: list[dict[str, str]], checkpoint_hash: str, name: str, required: bool
) -> None:
    checkpoints = {row.get("checkpoint_sha256", "") for row in rows if row.get("checkpoint_sha256")}
    datasets = {row.get("dataset_sha256", "") for row in rows if row.get("dataset_sha256")}
    audit.check(
        f"{name}.checkpoint_hash",
        checkpoints == {checkpoint_hash} if required else checkpoints in (set(), {checkpoint_hash}),
        f"expected only {checkpoint_hash}, found {sorted(checkpoints)}",
    )
    audit.check(
        f"{name}.dataset_hash",
        datasets == {DATASET_SHA256} if required else datasets in (set(), {DATASET_SHA256}),
        f"expected only {DATASET_SHA256}, found {sorted(datasets)}",
    )


def verify_phase1(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    audit = Audit()
    seed_reports: dict[str, dict[str, str]] = {}
    reference_paths: dict[str, list[str]] = {}
    manifest_hashes: dict[str, str] | None = None
    recomputed: dict[int, dict[str, dict[str, float]]] = {}

    for relative in SUMMARY_ARTIFACTS:
        audit.check("summary_artifacts", (root / relative).is_file(), f"missing {root / relative}")

    for seed in SEEDS:
        key = f"seed_{seed}"
        seed_dir = root / "runs" / key
        audit.check(f"{key}.directory", seed_dir.is_dir(), f"missing {seed_dir}")
        missing = [item for item in SEED_ARTIFACTS if not (seed_dir / item).is_file()]
        qualitative = seed_dir / "rapt_evaluation/qualitative/qualitative_sheet_index.csv"
        if seed == 2026 and not qualitative.is_file():
            missing.append("rapt_evaluation/qualitative/qualitative_sheet_index.csv")
        audit.check(f"{key}.artifacts", not missing, f"missing artifacts: {missing}")
        if not seed_dir.is_dir():
            continue

        complete = audit.json(seed_dir / "COMPLETE.json", f"{key}.complete")
        audit.check(
            f"{key}.complete",
            isinstance(complete, dict)
            and complete.get("status") == "PASS"
            and complete.get("seed") == seed
            and complete.get("test_access_protocol") == "strict",
            f"invalid completion marker: {complete!r}",
        )
        try:
            rapt_hash = sha256(seed_dir / "rapt_full/best.pt")
            resnet_hash = sha256(seed_dir / "resnet_training/B2_resnet18_pretrained/best.pt")
        except OSError as error:
            audit.check(f"{key}.checkpoint_files", False, str(error))
            continue
        seed_reports[str(seed)] = {
            "rapt_checkpoint_sha256": rapt_hash, "resnet_checkpoint_sha256": resnet_hash
        }
        for metric_model, display_model, expected_hash in (
            ("rahasrapt", "RAHAS-RAPT", rapt_hash),
            ("resnet18", "ResNet-18", resnet_hash),
        ):
            summary_metric = audit.json(
                root / "seed_level_metrics" / f"{metric_model}_seed{seed}.json",
                f"{key}.{metric_model}_summary_metric",
            )
            audit.check(
                f"{key}.{metric_model}_summary_metric_hashes",
                isinstance(summary_metric, dict)
                and summary_metric.get("model") == display_model
                and summary_metric.get("seed") == seed
                and summary_metric.get("checkpoint_sha256") == expected_hash
                and summary_metric.get("dataset_sha256") == DATASET_SHA256,
                "seed-level metric metadata mismatch",
            )
        rapt_metrics = audit.json(seed_dir / "rapt_evaluation/metrics.json", f"{key}.rapt_metrics")
        resnet_metrics = audit.json(
            seed_dir / "resnet_training/B2_resnet18_pretrained/metrics.json", f"{key}.resnet_metrics"
        )
        for model, value, expected_hash in (
            ("rapt", rapt_metrics, rapt_hash), ("resnet", resnet_metrics, resnet_hash)
        ):
            valid = isinstance(value, dict)
            audit.check(f"{key}.{model}_status", valid and value.get("status") == "PASS", "status mismatch")
            audit.check(f"{key}.{model}_seed", valid and value.get("seed") == seed, f"seed is not {seed}")
            audit.check(
                f"{key}.{model}_dataset_hash",
                valid and value.get("dataset_sha256") == DATASET_SHA256,
                "dataset hash mismatch",
            )
            audit.check(
                f"{key}.{model}_checkpoint_hash",
                valid and value.get("checkpoint_sha256") == expected_hash and _hash(expected_hash),
                "checkpoint hash mismatch",
            )
            if valid:
                _required_metrics(audit, value.get("validation"), f"{key}.{model}.validation_metrics")
                _required_metrics(audit, value.get("test"), f"{key}.{model}.test_metrics")

        if isinstance(rapt_metrics, dict):
            current = rapt_metrics.get("hashes")
            valid_hashes = (
                isinstance(current, dict)
                and set(current) == {"class_map", "train_manifest", "validation_manifest", "test_manifest"}
                and all(_hash(value) for value in current.values())
            )
            audit.check(f"{key}.manifest_hashes", valid_hashes, f"invalid hashes: {current!r}")
            if valid_hashes:
                if manifest_hashes is None:
                    manifest_hashes = current
                else:
                    audit.check(
                        f"{key}.manifest_hash_consistency",
                        current == manifest_hashes,
                        "hashes differ across seeds",
                    )

        _selection(audit, seed_dir / "rapt_warmup/selection_summary.json", f"{key}.warmup_selection")
        _selection(audit, seed_dir / "rapt_full/selection_summary.json", f"{key}.full_selection")
        _selection(audit, seed_dir / "rapt_evaluation/router_selection.json", f"{key}.router_selection")
        for label, relative in (
            ("warmup_epochs", "rapt_warmup/epoch_metrics.csv"),
            ("full_epochs", "rapt_full/epoch_metrics.csv"),
            ("resnet_epochs", "resnet_training/B2_resnet18_pretrained/epoch_metrics.csv"),
        ):
            _selection(audit, seed_dir / relative, f"{key}.{label}", csv_file=True)

        recomputed[seed] = {}
        source: dict[tuple[str, str], list[dict[str, str]]] = {}
        specs = (
            ("rapt", True, seed_dir / "rapt_evaluation", ("validation_predictions.csv", "test_predictions.csv")),
            ("resnet", False, seed_dir / "resnet_training/B2_resnet18_pretrained",
             ("predictions_validation.csv", "predictions_test.csv")),
        )
        for model, is_rapt, directory, filenames in specs:
            for split, filename in zip(("validation", "test"), filenames):
                name = f"{key}.{model}.{split}_predictions"
                rows = audit.csv(directory / filename, name)
                if rows is None:
                    continue
                source[(model, split)] = rows
                try:
                    paths = [_path(row["sample_path"]) for row in rows]
                    audit.check(f"{name}.nonempty", bool(rows), "file is empty")
                    audit.check(f"{name}.unique_paths", len(paths) == len(set(paths)), "paths are not unique")
                    audit.check(f"{name}.split", all(row.get("split") == split for row in rows), "split mismatch")
                    reference = reference_paths.setdefault(split, paths)
                    audit.check(f"{name}.path_alignment", paths == reference, "paths differ from reference")
                    _row_hashes(audit, rows, rapt_hash if is_rapt else resnet_hash, name, required=is_rapt)
                    recomputed[seed][model if split == "test" else f"{model}_validation"] = (
                        metrics_from_predictions(rows, is_rapt)
                    )
                except (KeyError, TypeError, ValueError) as error:
                    audit.check(f"{name}.schema", False, str(error))

        for split in ("validation", "test"):
            left, right = source.get(("rapt", split)), source.get(("resnet", split))
            if left is not None and right is not None:
                left_paths = [_path(row.get("sample_path", "")) for row in left]
                right_paths = [_path(row.get("sample_path", "")) for row in right]
                audit.check(
                    f"{key}.{split}_model_alignment",
                    len(left) == len(right) and left_paths == right_paths,
                    f"RAPT/ResNet rows or paths differ ({len(left)} versus {len(right)})",
                )
        validation_paths = {_path(row.get("sample_path", "")) for row in source.get(("rapt", "validation"), [])}
        test_paths = {_path(row.get("sample_path", "")) for row in source.get(("rapt", "test"), [])}
        audit.check(
            f"{key}.split_disjoint_paths",
            bool(validation_paths) and bool(test_paths) and validation_paths.isdisjoint(test_paths),
            "validation/test paths overlap or are empty",
        )

        for model, file_model, expected_hash in (
            ("rapt", "rapt", rapt_hash), ("resnet", "resnet18", resnet_hash)
        ):
            for split in ("validation", "test"):
                name = f"{key}.{model}.{split}_summary_copy"
                copied = audit.csv(
                    root / "seed_level_predictions" / f"{file_model}_seed{seed}_{split}.csv", name
                )
                original = source.get((model, split))
                if copied is None or original is None:
                    continue
                copied_paths = [_path(row.get("sample_path", "")) for row in copied]
                original_paths = [_path(row.get("sample_path", "")) for row in original]
                audit.check(
                    f"{name}.alignment",
                    len(copied) == len(original) and copied_paths == original_paths,
                    "summary copy differs from source",
                )
                _row_hashes(audit, copied, expected_hash, name, required=True)
                audit.check(
                    f"{name}.seed",
                    all(row.get("model_seed") == str(seed) for row in copied),
                    "model_seed mismatch",
                )

    paired = audit.csv(root / "paired_comparison.csv", "paired_summary")
    if paired is not None:
        counts = Counter(row.get("seed") for row in paired)
        audit.check(
            "paired_summary.seeds",
            len(paired) == len(SEEDS) and counts == Counter(map(str, SEEDS)),
            f"expected one row per seed, found {dict(counts)}",
        )
        for row in paired:
            try:
                seed = int(row["seed"])
                rapt, resnet = recomputed[seed]["rapt"], recomputed[seed]["resnet"]
            except (KeyError, TypeError, ValueError):
                audit.check("paired_summary.schema", False, f"invalid or unavailable row: {row!r}")
                continue
            for metric in METRICS:
                expected = (rapt[metric], resnet[metric], rapt[metric] - resnet[metric])
                for field, value in zip(("rapt", "resnet", "difference"), expected):
                    audit.check(
                        f"paired_summary.seed_{seed}.{metric}.{field}",
                        _close(row.get(f"{field}_{metric}"), value),
                        f"does not equal recomputed {value}",
                    )

    repeated = audit.csv(root / "repeated_seed_summary.csv", "repeated_seed_summary")
    if repeated is not None:
        found = Counter((row.get("seed"), row.get("model")) for row in repeated)
        expected = Counter((str(seed), model) for seed in SEEDS for model in ("RAHAS-RAPT", "ResNet-18"))
        audit.check("repeated_seed_summary.rows", found == expected, f"unexpected rows: {dict(found)}")
        for row in repeated:
            try:
                seed = int(row["seed"])
                model = "rapt" if row["model"] == "RAHAS-RAPT" else "resnet"
                values = recomputed[seed][model]
                expected_hash = seed_reports[str(seed)][f"{model}_checkpoint_sha256"]
            except (KeyError, TypeError, ValueError):
                continue
            audit.check(
                f"repeated_seed_summary.seed_{seed}.{model}.hashes",
                row.get("checkpoint_sha256") == expected_hash
                and row.get("dataset_sha256") == DATASET_SHA256,
                "summary hashes do not match source artifacts",
            )
            for metric in METRICS:
                audit.check(
                    f"repeated_seed_summary.seed_{seed}.{model}.{metric}",
                    _close(row.get(f"test_{metric}"), values[metric]),
                    f"does not equal recomputed {values[metric]}",
                )

    statistical = audit.json(root / "statistical_summary.json", "statistical_summary")
    audit.check(
        "statistical_summary.dataset_hash",
        isinstance(statistical, dict) and statistical.get("dataset_sha256") == DATASET_SHA256,
        "dataset hash mismatch",
    )
    return {
        "status": "PASS" if not audit.errors else "FAIL",
        "root": str(root),
        "dataset_sha256": DATASET_SHA256,
        "seeds": seed_reports,
        "checks": audit.checks,
        "errors": audit.errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently verify strict five-seed RAPT Phase-1 outputs.")
    parser.add_argument("--root", type=Path, default=Path("pipeline/experiments/rahas_rapt_validation_v1"))
    parser.add_argument("--output", type=Path, help="Optional JSON report path; stdout is always written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify_phase1(args.root)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
