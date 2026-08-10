from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_rahas_rapt_phase1_v1 import classification_metrics, read_csv, sha256, write_csv


SEEDS = (2026, 17, 42, 123, 3407)
DATASET_SHA256 = "4d241e39f754b8cb4271eb94194eb07a706d50ccf61cf966063e87f91b0a8d7b"
FREQUENCY_BINS = ("one_shot", "2_4", "5_9", "10_19", "20_plus", "10_plus")
CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "top3",
    "top5",
)
COMPONENT_METRICS = ("base_accuracy", "modifier_accuracy")
FREQUENCY_METRICS = tuple(f"{name}_accuracy" for name in FREQUENCY_BINS)
METRICS = CLASSIFICATION_METRICS + COMPONENT_METRICS + FREQUENCY_METRICS
CO_PRIMARY_METRICS = ("accuracy", "macro_f1")
T_CRITICAL_95_DF4 = 2.7764451051977987
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the strict five-seed RAPT Phase-1 comparison.")
    parser.add_argument(
        "--root", type=Path, default=Path("pipeline/experiments/rahas_rapt_validation_v1")
    )
    return parser.parse_args()


def bool_value(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _finite_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite: {value!r}")
    return result


def _integer(value: object, field: str) -> int:
    number = _finite_float(value, field)
    result = int(number)
    if number != result:
        raise ValueError(f"{field} must be an integer: {value!r}")
    return result


def _normalized_path(value: object) -> str:
    return str(value).replace("\\", "/")


def _field(row: dict[str, str], name: str, context: str) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"{context}: missing {name}")
    return str(value)


def _model_keys(rapt: bool) -> dict[str, str]:
    if rapt:
        return {
            "target_index": "true_class_index",
            "target_label": "true_full_label",
            "prediction_label": "predicted_full_label",
            "true_base": "true_base_label",
            "true_modifier": "true_modifier_label",
            "derived_base": "predicted_base_from_full",
            "derived_modifier": "predicted_modifier_from_full",
            "count": "training_original_sample_count",
        }
    return {
        "target_index": "class_index",
        "target_label": "class_label",
        "prediction_label": "predicted_label",
        "true_base": "true_base",
        "true_modifier": "true_modifier",
        "derived_base": "predicted_base_from_character",
        "derived_modifier": "predicted_modifier_from_character",
        "count": "training_original_count",
    }


def _fine_frequency_bin(count: int) -> str:
    if count == 1:
        return "one_shot"
    if 2 <= count <= 4:
        return "2_4"
    if 5 <= count <= 9:
        return "5_9"
    if 10 <= count <= 19:
        return "10_19"
    if count >= 20:
        return "20_plus"
    raise ValueError(f"training count must be positive, found {count}")


def validate_prediction_rows(
    rows: list[dict[str, str]],
    rapt: bool,
    *,
    context: str,
    seed: int | None = None,
    split: str | None = None,
    checkpoint_hash: str | None = None,
) -> None:
    if not rows:
        raise ValueError(f"{context}: prediction CSV is empty")
    keys = _model_keys(rapt)
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        where = f"{context}: row {index}"
        sample_path = _normalized_path(_field(row, "sample_path", where))
        sample_id = _field(row, "sample_id", where)
        identity = (sample_path, sample_id)
        if identity in identities:
            raise ValueError(f"{where}: duplicate sample identity {identity!r}")
        identities.add(identity)

        target = _integer(_field(row, keys["target_index"], where), f"{where} target index")
        prediction = _integer(
            _field(row, "predicted_class_index", where), f"{where} predicted_class_index"
        )
        if target < 0 or prediction < 0:
            raise ValueError(f"{where}: class indices must be non-negative")
        target_label = _field(row, keys["target_label"], where)
        prediction_label = _field(row, keys["prediction_label"], where)
        _field(row, keys["true_base"], where)
        _field(row, keys["true_modifier"], where)
        _field(row, keys["derived_base"], where)
        _field(row, keys["derived_modifier"], where)
        if rapt:
            _field(row, "predicted_base_label", where)
            _field(row, "predicted_modifier_label", where)

        count = _integer(_field(row, keys["count"], where), f"{where} training count")
        expected_bin = _fine_frequency_bin(count)
        supplied_bin = _field(row, "frequency_bin", where)
        allowed_bins = {expected_bin}
        if count >= 10:
            allowed_bins.add("10_plus")
        if supplied_bin not in allowed_bins:
            raise ValueError(
                f"{where}: frequency_bin {supplied_bin!r} is inconsistent with training count {count}"
            )

        scores = []
        labels = []
        for rank in range(1, 6):
            labels.append(_field(row, f"top_{rank}_label", where))
            score = _finite_float(
                _field(row, f"top_{rank}_score", where), f"{where} top_{rank}_score"
            )
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{where}: top_{rank}_score must be in [0, 1]")
            scores.append(score)
        if len(set(labels)) != 5:
            raise ValueError(f"{where}: top-5 labels must be distinct")
        if labels[0] != prediction_label:
            raise ValueError(f"{where}: top_1_label does not match the predicted full label")
        if any(left + TOLERANCE < right for left, right in zip(scores, scores[1:])):
            raise ValueError(f"{where}: top-5 scores are not monotonically non-increasing")
        if bool_value(row.get("correct")) != (target == prediction):
            raise ValueError(f"{where}: correct is inconsistent with target/prediction indices")
        if target_label == prediction_label and target != prediction:
            raise ValueError(f"{where}: equal full labels have different class indices")

        if split is not None and row.get("split") != split:
            raise ValueError(f"{where}: split {row.get('split')!r} does not match {split!r}")
        expected_provenance = {
            "model_seed": str(seed) if seed is not None else None,
            "checkpoint_sha256": checkpoint_hash,
            "dataset_sha256": DATASET_SHA256,
        }
        for field, expected in expected_provenance.items():
            supplied = row.get(field)
            if expected is not None and supplied not in (None, "") and str(supplied) != expected:
                raise ValueError(
                    f"{where}: {field} provenance mismatch; expected {expected!r}, found {supplied!r}"
                )


def _alignment_signature(row: dict[str, str], rapt: bool) -> tuple[object, ...]:
    keys = _model_keys(rapt)
    return (
        _normalized_path(row["sample_path"]),
        row["sample_id"],
        _integer(row[keys["target_index"]], "target index"),
        row[keys["target_label"]],
        row[keys["true_base"]],
        row[keys["true_modifier"]],
        _integer(row[keys["count"]], "training count"),
    )


def require_exact_model_alignment(
    rapt_rows: list[dict[str, str]], resnet_rows: list[dict[str, str]], *, context: str
) -> None:
    if len(rapt_rows) != len(resnet_rows):
        raise ValueError(
            f"{context}: RAPT/ResNet row count mismatch ({len(rapt_rows)} versus {len(resnet_rows)})"
        )
    for offset, (rapt_row, resnet_row) in enumerate(zip(rapt_rows, resnet_rows), start=2):
        left = _alignment_signature(rapt_row, True)
        right = _alignment_signature(resnet_row, False)
        if left != right:
            raise ValueError(
                f"{context}: RAPT/ResNet sample or target mismatch at row {offset}: {left!r} != {right!r}"
            )


def metrics_from_rows(rows: list[dict[str, str]], rapt: bool) -> dict[str, float]:
    validate_prediction_rows(rows, rapt, context="metrics")
    keys = _model_keys(rapt)
    target = np.asarray([_integer(row[keys["target_index"]], "target index") for row in rows])
    prediction = np.asarray(
        [_integer(row["predicted_class_index"], "predicted class index") for row in rows]
    )
    top_hits = np.asarray(
        [
            [row[f"top_{rank}_label"] == row[keys["target_label"]] for rank in range(1, 6)]
            for row in rows
        ],
        dtype=bool,
    )
    metrics, _ = classification_metrics(prediction, target, None, None)
    metrics["top3"] = float(np.mean(np.any(top_hits[:, :3], axis=1)))
    metrics["top5"] = float(np.mean(np.any(top_hits, axis=1)))
    metrics["base_accuracy"] = float(
        np.mean([row[keys["true_base"]] == row[keys["derived_base"]] for row in rows])
    )
    metrics["modifier_accuracy"] = float(
        np.mean([row[keys["true_modifier"]] == row[keys["derived_modifier"]] for row in rows])
    )
    if rapt:
        metrics["auxiliary_base_accuracy"] = float(
            np.mean([row["true_base_label"] == row["predicted_base_label"] for row in rows])
        )
        metrics["auxiliary_modifier_accuracy"] = float(
            np.mean(
                [row["true_modifier_label"] == row["predicted_modifier_label"] for row in rows]
            )
        )

    fine_bins: dict[str, list[bool]] = defaultdict(list)
    for row, correct in zip(rows, prediction == target):
        fine_bins[_fine_frequency_bin(_integer(row[keys["count"]], "training count"))].append(
            bool(correct)
        )
    required_fine = FREQUENCY_BINS[:-1]
    missing = [name for name in required_fine if not fine_bins[name]]
    if missing:
        raise ValueError(f"metrics: frequency bins have no samples: {missing}")
    for name in required_fine:
        metrics[f"{name}_accuracy"] = float(np.mean(fine_bins[name]))
        metrics[f"{name}_samples"] = float(len(fine_bins[name]))
    ten_plus = fine_bins["10_19"] + fine_bins["20_plus"]
    metrics["10_plus_accuracy"] = float(np.mean(ten_plus))
    metrics["10_plus_samples"] = float(len(ten_plus))

    required = ("samples", "represented_classes") + METRICS
    absent = [name for name in required if name not in metrics]
    if absent:
        raise ValueError(f"metrics: incomplete metric set: {absent}")
    for name, value in metrics.items():
        _finite_float(value, f"metric {name}")
    return metrics


def _prepared_prediction_copy(
    rows: list[dict[str, str]], seed: int, model: str, checkpoint_hash: str
) -> list[dict[str, Any]]:
    copies: list[dict[str, Any]] = []
    for row_number, source_row in enumerate(rows, start=2):
        row: dict[str, Any] = dict(source_row)
        expected = {
            "model": model,
            "model_seed": str(seed),
            "checkpoint_sha256": checkpoint_hash,
            "dataset_sha256": DATASET_SHA256,
        }
        for field, value in expected.items():
            supplied = row.get(field)
            if supplied not in (None, "") and str(supplied) != value:
                raise ValueError(
                    f"prediction row {row_number}: {field} provenance mismatch; "
                    f"expected {value!r}, found {supplied!r}"
                )
            row[field] = value
        copies.append(row)
    return copies


def copy_prediction(
    source: Path,
    destination: Path,
    seed: int,
    model: str,
    checkpoint: Path,
    *,
    rapt: bool | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    rows = read_csv(source)
    is_rapt = model == "RAHAS-RAPT" if rapt is None else rapt
    checkpoint_hash = sha256(checkpoint)
    validate_prediction_rows(
        rows,
        is_rapt,
        context=str(source),
        seed=seed,
        split=split,
        checkpoint_hash=checkpoint_hash,
    )
    copies = _prepared_prediction_copy(rows, seed, model, checkpoint_hash)
    write_csv(destination, copies)
    return copies


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot describe an empty sample")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("descriptive statistics require finite values")
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half_width = (
        T_CRITICAL_95_DF4 * standard_deviation / math.sqrt(len(array))
        if len(array) == 5
        else 0.0
    )
    mean = float(array.mean())
    return {
        "n": float(len(array)),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def bootstrap_interval(
    values: list[float], seed: int = 2026, samples: int = 100000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap requires a non-empty finite sample")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def _descending_average_rank(values: list[float], selected: float) -> float:
    greater = sum(value > selected and not math.isclose(value, selected) for value in values)
    tied = sum(math.isclose(value, selected, rel_tol=1e-12, abs_tol=1e-12) for value in values)
    return 1.0 + greater + (tied - 1) / 2.0


def paired_metric_statistics(
    metric: str, paired_rows: list[dict[str, Any]], *, bootstrap_samples: int = 100000
) -> dict[str, Any]:
    differences = [_finite_float(row[f"difference_{metric}"], metric) for row in paired_rows]
    if len(differences) != len(SEEDS):
        raise ValueError(f"{metric}: expected {len(SEEDS)} paired seeds, found {len(differences)}")
    seeds = [_integer(row["seed"], "seed") for row in paired_rows]
    if tuple(seeds) != SEEDS:
        raise ValueError(f"{metric}: seeds must be in preregistered order {SEEDS}, found {tuple(seeds)}")
    array = np.asarray(differences, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    standard_error = sd / math.sqrt(len(array))
    t_half_width = T_CRITICAL_95_DF4 * standard_error
    bootstrap_low, bootstrap_high = bootstrap_interval(
        differences, seed=2026, samples=bootstrap_samples
    )
    selected = differences[seeds.index(2026)]
    median = float(np.median(array))
    return {
        "metric": metric,
        "n_pairs": len(array),
        "degrees_of_freedom": len(array) - 1,
        "per_seed_differences": dict(zip(map(str, seeds), differences)),
        "positive_seeds": int(np.sum(array > 0)),
        "negative_seeds": int(np.sum(array < 0)),
        "tied_seeds": int(np.sum(array == 0)),
        "mean_difference": mean,
        "standard_deviation_difference": sd,
        "standard_error_difference": standard_error,
        "paired_t_statistic": mean / standard_error if standard_error > 0 else None,
        "paired_t_df": len(array) - 1,
        "paired_t_ci95_low": mean - t_half_width,
        "paired_t_ci95_high": mean + t_half_width,
        "bootstrap_method": "descriptive paired-seed bootstrap",
        "bootstrap_resamples": bootstrap_samples,
        "bootstrap_seed": 2026,
        "bootstrap_ci95_low": bootstrap_low,
        "bootstrap_ci95_high": bootstrap_high,
        "low_n_warning": "LOW-N: five seed pairs; bootstrap interval is descriptive, not confirmatory.",
        "seed2026_difference": selected,
        "seed2026_descending_rank": _descending_average_rank(differences, selected),
        "median_difference": median,
        "seed2026_signed_distance_from_median": selected - median,
        "seed2026_absolute_distance_from_median": abs(selected - median),
    }


def co_primary_verdict(paired_statistics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    endpoints = {metric: paired_statistics[metric] for metric in CO_PRIMARY_METRICS}

    def supports_advantage(value: dict[str, Any]) -> bool:
        return (
            value["mean_difference"] > 0
            and value["positive_seeds"] >= 4
            and value["paired_t_ci95_low"] > 0
        )

    def supports_disadvantage(value: dict[str, Any]) -> bool:
        return (
            value["mean_difference"] < 0
            and value["negative_seeds"] >= 4
            and value["paired_t_ci95_high"] < 0
        )

    advantage = all(supports_advantage(value) for value in endpoints.values())
    disadvantage = all(supports_disadvantage(value) for value in endpoints.values())
    if advantage:
        verdict = "RAPT advantage supported on both co-primary endpoints"
        recommendation = "Recommend RAHAS-RAPT as the primary recognition candidate."
    elif disadvantage:
        verdict = "RAPT disadvantage supported on both co-primary endpoints"
        recommendation = "Do not recommend RAHAS-RAPT as the primary recognition candidate."
    else:
        verdict = "Uncertain: co-primary evidence does not establish a RAPT advantage"
        recommendation = (
            "Do not recommend RAHAS-RAPT as the primary recognition candidate from Phase-1 evidence."
        )
    return {
        "verdict": verdict,
        "recommend_rapt_as_primary": advantage,
        "recommendation": recommendation,
        "preregistered_rule": (
            "Both accuracy and macro_f1 must have a positive mean difference, RAPT wins on at "
            "least 4/5 seeds, and the paired t(4) 95% CI is entirely above zero. The descriptive "
            "low-n bootstrap is reported but is not a decision criterion."
        ),
    }


def _qualitative_rows(
    rapt_rows: list[dict[str, str]], resnet_rows: list[dict[str, str]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    require_exact_model_alignment(rapt_rows, resnet_rows, context=f"seed {seed} qualitative")
    groups: dict[str, list[dict[str, Any]]] = {
        "rapt_correct_resnet_wrong": [],
        "resnet_correct_rapt_wrong": [],
        "both_wrong": [],
        "both_correct": [],
    }
    for rapt_row, resnet_row in zip(rapt_rows, resnet_rows):
        rapt_correct = bool_value(rapt_row["correct"])
        resnet_correct = bool_value(resnet_row["correct"])
        if rapt_correct and not resnet_correct:
            category = "rapt_correct_resnet_wrong"
        elif resnet_correct and not rapt_correct:
            category = "resnet_correct_rapt_wrong"
        elif rapt_correct:
            category = "both_correct"
        else:
            category = "both_wrong"
        groups[category].append(
            {
                "seed": seed,
                "comparison": category,
                "sample_path": rapt_row["sample_path"],
                "sample_id": rapt_row["sample_id"],
                "target_full_label": rapt_row["true_full_label"],
                "target_class_index": rapt_row["true_class_index"],
                "rapt_predicted_full_label": rapt_row["predicted_full_label"],
                "rapt_predicted_class_index": rapt_row["predicted_class_index"],
                "rapt_confidence": rapt_row.get("top_1_score", ""),
                "resnet_predicted_full_label": resnet_row["predicted_label"],
                "resnet_predicted_class_index": resnet_row["predicted_class_index"],
                "resnet_confidence": resnet_row.get("top_1_score", ""),
            }
        )
    return groups


def generate_cross_model_qualitative_sheets(
    rapt_rows: list[dict[str, str]],
    resnet_rows: list[dict[str, str]],
    output: Path,
    *,
    seed: int = 2026,
    page_size: int = 48,
) -> list[dict[str, Any]]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    groups = _qualitative_rows(rapt_rows, resnet_rows, seed)
    index_rows: list[dict[str, Any]] = []
    tile_width, tile_height, columns = 190, 160, 6
    for category, rows in groups.items():
        for page, start in enumerate(range(0, len(rows), page_size), start=1):
            subset = rows[start : start + page_size]
            canvas = Image.new(
                "RGB",
                (tile_width * columns, tile_height * math.ceil(len(subset) / columns)),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            sheet_name = f"{category}_{page:03d}.png"
            for offset, row in enumerate(subset):
                tile_number = start + offset + 1
                source = Path(str(row["sample_path"]))
                if not source.is_absolute():
                    source = Path.cwd() / source
                if not source.is_file():
                    raise FileNotFoundError(f"qualitative source image does not exist: {source}")
                with Image.open(source) as image:
                    tile = ImageOps.contain(image.convert("L"), (150, 108))
                x = (offset % columns) * tile_width
                y = (offset // columns) * tile_height
                canvas.paste(tile.convert("RGB"), (x + (tile_width - tile.width) // 2, y + 42))
                draw.text((x + 4, y + 4), f"#{tile_number:04d} T{row['target_class_index']}", fill="black")
                draw.text(
                    (x + 4, y + 21),
                    f"R{row['rapt_predicted_class_index']} N{row['resnet_predicted_class_index']}",
                    fill="black",
                )
                index_rows.append({"sheet": sheet_name, "tile": tile_number, **row})
            canvas.save(output / sheet_name)
    write_csv(output / "cross_model_qualitative_index.csv", index_rows)
    return index_rows


def _selected_best_epoch(path: Path) -> int:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path}: epoch metrics are empty")
    scores = [_finite_float(row.get("validation_macro_f1"), f"{path}: validation_macro_f1") for row in rows]
    selected = [index for index, row in enumerate(rows) if bool_value(row.get("best"))]
    if len(selected) != 1:
        raise ValueError(f"{path}: expected exactly one best=true row, found {len(selected)}")
    index = selected[0]
    if not math.isclose(scores[index], max(scores), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{path}: best=true row does not maximize validation_macro_f1")
    return _integer(rows[index].get("epoch"), f"{path}: epoch")


def _validate_complete_marker(path: Path, seed: int) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read completion marker {path}: {error}") from error
    expected = {"status": "PASS", "seed": seed, "test_access_protocol": "strict"}
    for field, required in expected.items():
        if value.get(field) != required:
            raise ValueError(
                f"{path}: {field} must be {required!r}, found {value.get(field)!r}"
            )


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _fmt(summary: dict[str, Any], metric: str) -> str:
    value = summary[metric]
    return f"{value['mean']:.4f} +/- {value['standard_deviation']:.4f}"


def _report(
    statistics: dict[str, Any], paired_rows: list[dict[str, Any]], qualitative_rows: list[dict[str, Any]]
) -> str:
    models = statistics["models"]
    paired = statistics["paired"]
    decision = statistics["decision"]
    frequency_lines = []
    for model in ("RAHAS-RAPT", "ResNet-18"):
        frequency_lines.append(
            f"| {model} | "
            + " | ".join(_fmt(models[model], f"{name}_accuracy") for name in FREQUENCY_BINS)
            + " |"
        )
    diagnostic_lines = []
    for metric in METRICS:
        item = paired[metric]
        diagnostic_lines.append(
            f"| {metric} | {item['seed2026_difference']:.4f} | "
            f"{item['seed2026_descending_rank']:.1f}/5 | {item['median_difference']:.4f} | "
            f"{item['seed2026_signed_distance_from_median']:.4f} |"
        )
    comparison_counts: dict[str, int] = defaultdict(int)
    for row in qualitative_rows:
        comparison_counts[str(row["comparison"])] += 1
    qualitative_summary = ", ".join(
        f"{name}={comparison_counts.get(name, 0)}"
        for name in (
            "rapt_correct_resnet_wrong",
            "resnet_correct_rapt_wrong",
            "both_wrong",
            "both_correct",
        )
    )
    accuracy = paired["accuracy"]
    macro = paired["macro_f1"]
    return f"""# RAHAS-RAPT Validation Phase 1 Report

## Protocol and integrity

- Frozen dataset SHA-256: `{DATASET_SHA256}`
- Preregistered seeds: {', '.join(map(str, SEEDS))}
- Checkpoint selection: validation macro-F1 only; test predictions are summarized only after strict completion markers.
- Evidence validation: source provenance is checked before copying, all metrics and top-5 scores must be finite and complete, top-5 scores must be non-increasing, and RAPT/ResNet sample IDs, paths, targets, components, and training counts must align exactly.
- Cross-model component metrics are derived from each model's predicted full label. RAPT auxiliary-head component metrics are reported separately and are not used as the ResNet comparison basis.

## Main test results

Values are mean +/- sample SD across five paired seeds.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Top-3 | Top-5 | Base from full | Modifier from full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAHAS-RAPT | {_fmt(models['RAHAS-RAPT'], 'accuracy')} | {_fmt(models['RAHAS-RAPT'], 'balanced_accuracy')} | {_fmt(models['RAHAS-RAPT'], 'macro_f1')} | {_fmt(models['RAHAS-RAPT'], 'weighted_f1')} | {_fmt(models['RAHAS-RAPT'], 'top3')} | {_fmt(models['RAHAS-RAPT'], 'top5')} | {_fmt(models['RAHAS-RAPT'], 'base_accuracy')} | {_fmt(models['RAHAS-RAPT'], 'modifier_accuracy')} |
| ResNet-18 | {_fmt(models['ResNet-18'], 'accuracy')} | {_fmt(models['ResNet-18'], 'balanced_accuracy')} | {_fmt(models['ResNet-18'], 'macro_f1')} | {_fmt(models['ResNet-18'], 'weighted_f1')} | {_fmt(models['ResNet-18'], 'top3')} | {_fmt(models['ResNet-18'], 'top5')} | {_fmt(models['ResNet-18'], 'base_accuracy')} | {_fmt(models['ResNet-18'], 'modifier_accuracy')} |

RAPT auxiliary heads: base accuracy {_fmt(models['RAHAS-RAPT'], 'auxiliary_base_accuracy')}; modifier accuracy {_fmt(models['RAHAS-RAPT'], 'auxiliary_modifier_accuracy')}.

## Frequency-stratified accuracy

| Model | One-shot | 2-4 | 5-9 | 10-19 | 20+ | 10+ combined |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(frequency_lines)}

## Paired seed analysis

- Accuracy difference (RAPT - ResNet): mean {accuracy['mean_difference']:.4f}; RAPT wins {accuracy['positive_seeds']}/5 seeds; paired t(4) 95% CI [{accuracy['paired_t_ci95_low']:.4f}, {accuracy['paired_t_ci95_high']:.4f}].
- Macro-F1 difference: mean {macro['mean_difference']:.4f}; RAPT wins {macro['positive_seeds']}/5 seeds; paired t(4) 95% CI [{macro['paired_t_ci95_low']:.4f}, {macro['paired_t_ci95_high']:.4f}].
- Descriptive paired-seed bootstrap accuracy interval: [{accuracy['bootstrap_ci95_low']:.4f}, {accuracy['bootstrap_ci95_high']:.4f}]. **LOW-N: five seed pairs; descriptive only, not confirmatory.**
- Descriptive paired-seed bootstrap macro-F1 interval: [{macro['bootstrap_ci95_low']:.4f}, {macro['bootstrap_ci95_high']:.4f}]. **LOW-N: five seed pairs; descriptive only, not confirmatory.**
- Full per-seed differences are in `paired_comparison.csv`; complete paired inference and diagnostics are in `paired_statistics.csv`.

## Seed 2026 representativeness

Ranks order paired differences from largest (rank 1) to smallest; ties receive average ranks.

| Metric | Seed 2026 difference | Rank | Five-seed median | Signed distance from median |
|---|---:|---:|---:|---:|
{chr(10).join(diagnostic_lines)}

## Qualitative comparison

Cross-model sheets were regenerated directly from aligned saved test predictions for representative seed 2026. Counts: {qualitative_summary}. See `cross_model_qualitative/seed_2026/cross_model_qualitative_index.csv`.

## Preregistered co-primary verdict

Decision rule: {decision['preregistered_rule']}

**{decision['verdict']}**

{decision['recommendation']}

## Final Phase 1 verdict

**{statistics['phase1_verdict']}**

This Phase-1 decision is based on accuracy and macro-F1 jointly. Secondary balanced, weighted, top-k, component, and frequency-stratified metrics are descriptive and cannot override a failed or uncertain co-primary decision.
"""


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    bundles: dict[int, dict[str, dict[str, list[dict[str, str]]]]] = {}
    checkpoint_hashes: dict[int, dict[str, str]] = {}
    best_epochs: dict[int, dict[str, int]] = {}

    for seed in SEEDS:
        seed_dir = root / "runs" / f"seed_{seed}"
        _validate_complete_marker(seed_dir / "COMPLETE.json", seed)
        rapt_checkpoint = seed_dir / "rapt_full" / "best.pt"
        resnet_dir = seed_dir / "resnet_training" / "B2_resnet18_pretrained"
        resnet_checkpoint = resnet_dir / "best.pt"
        hashes = {"RAHAS-RAPT": sha256(rapt_checkpoint), "ResNet-18": sha256(resnet_checkpoint)}
        checkpoint_hashes[seed] = hashes
        best_epochs[seed] = {
            "RAHAS-RAPT": _selected_best_epoch(seed_dir / "rapt_full" / "epoch_metrics.csv"),
            "ResNet-18": _selected_best_epoch(resnet_dir / "epoch_metrics.csv"),
        }
        bundles[seed] = {"RAHAS-RAPT": {}, "ResNet-18": {}}
        for split, rapt_name, resnet_name in (
            ("validation", "validation_predictions.csv", "predictions_validation.csv"),
            ("test", "test_predictions.csv", "predictions_test.csv"),
        ):
            rapt_rows = read_csv(seed_dir / "rapt_evaluation" / rapt_name)
            resnet_rows = read_csv(resnet_dir / resnet_name)
            validate_prediction_rows(
                rapt_rows,
                True,
                context=f"seed {seed} RAPT {split}",
                seed=seed,
                split=split,
                checkpoint_hash=hashes["RAHAS-RAPT"],
            )
            validate_prediction_rows(
                resnet_rows,
                False,
                context=f"seed {seed} ResNet {split}",
                seed=seed,
                split=split,
                checkpoint_hash=hashes["ResNet-18"],
            )
            require_exact_model_alignment(
                rapt_rows, resnet_rows, context=f"seed {seed} {split}"
            )
            bundles[seed]["RAHAS-RAPT"][split] = rapt_rows
            bundles[seed]["ResNet-18"][split] = resnet_rows

    prediction_dir = root / "seed_level_predictions"
    metric_dir = root / "seed_level_metrics"
    seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, list[float]]] = {
        "RAHAS-RAPT": defaultdict(list),
        "ResNet-18": defaultdict(list),
    }
    per_seed_metrics: dict[int, dict[str, dict[str, dict[str, float]]]] = {}

    for seed in SEEDS:
        per_seed_metrics[seed] = {}
        for model, file_model, rapt in (
            ("RAHAS-RAPT", "rapt", True),
            ("ResNet-18", "resnet18", False),
        ):
            per_seed_metrics[seed][model] = {}
            for split in ("validation", "test"):
                source_rows = bundles[seed][model][split]
                copied = _prepared_prediction_copy(
                    source_rows, seed, model, checkpoint_hashes[seed][model]
                )
                write_csv(prediction_dir / f"{file_model}_seed{seed}_{split}.csv", copied)
                per_seed_metrics[seed][model][split] = metrics_from_rows(source_rows, rapt)
            row: dict[str, Any] = {
                "model": model,
                "seed": seed,
                "best_epoch": best_epochs[seed][model],
                "checkpoint_sha256": checkpoint_hashes[seed][model],
                "dataset_sha256": DATASET_SHA256,
            }
            for split in ("validation", "test"):
                row.update(
                    {
                        f"{split}_{key}": value
                        for key, value in per_seed_metrics[seed][model][split].items()
                    }
                )
            seed_rows.append(row)
            for metric in METRICS:
                all_metrics[model][metric].append(per_seed_metrics[seed][model]["test"][metric])
            if model == "RAHAS-RAPT":
                for metric in ("auxiliary_base_accuracy", "auxiliary_modifier_accuracy"):
                    all_metrics[model][metric].append(
                        per_seed_metrics[seed][model]["test"][metric]
                    )
            _json_write(
                metric_dir / f"{model.lower().replace('-', '').replace(' ', '_')}_seed{seed}.json",
                row,
            )

        paired: dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            rapt_value = per_seed_metrics[seed]["RAHAS-RAPT"]["test"][metric]
            resnet_value = per_seed_metrics[seed]["ResNet-18"]["test"][metric]
            paired[f"rapt_{metric}"] = rapt_value
            paired[f"resnet_{metric}"] = resnet_value
            paired[f"difference_{metric}"] = rapt_value - resnet_value
        paired_rows.append(paired)

    write_csv(root / "repeated_seed_summary.csv", seed_rows)
    write_csv(root / "paired_comparison.csv", paired_rows)

    statistics: dict[str, Any] = {"models": {}, "paired": {}}
    descriptive_rows: list[dict[str, Any]] = []
    for model, metric_values in all_metrics.items():
        statistics["models"][model] = {}
        for metric, values in metric_values.items():
            summary = describe(values)
            statistics["models"][model][metric] = summary
            descriptive_rows.append({"model": model, "metric": metric, **summary})
    write_csv(root / "tables" / "descriptive_statistics.csv", descriptive_rows)

    paired_statistic_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        summary = paired_metric_statistics(metric, paired_rows)
        statistics["paired"][metric] = summary
        paired_statistic_rows.append(
            {key: value for key, value in summary.items() if key != "per_seed_differences"}
        )
    write_csv(root / "paired_statistics.csv", paired_statistic_rows)
    statistics["decision"] = co_primary_verdict(statistics["paired"])
    if statistics["decision"]["recommend_rapt_as_primary"]:
        statistics["phase1_verdict"] = "RAPT advantage stable"
    elif all(
        statistics["paired"][metric]["mean_difference"] > 0
        for metric in ("accuracy", "macro_f1")
    ):
        statistics["phase1_verdict"] = "RAPT advantage promising but uncertain"
    else:
        statistics["phase1_verdict"] = "RAPT advantage not stable"
    statistics["dataset_sha256"] = DATASET_SHA256
    statistics["seeds"] = list(SEEDS)

    qualitative_rows = generate_cross_model_qualitative_sheets(
        bundles[2026]["RAHAS-RAPT"]["test"],
        bundles[2026]["ResNet-18"]["test"],
        root / "cross_model_qualitative" / "seed_2026",
        seed=2026,
    )
    _json_write(root / "statistical_summary.json", statistics)
    (root / "PHASE1_VALIDATION_REPORT.md").write_text(
        _report(statistics, paired_rows, qualitative_rows), encoding="utf-8"
    )
    print(statistics["decision"]["verdict"])
    print(statistics["decision"]["recommendation"])


if __name__ == "__main__":
    main()
