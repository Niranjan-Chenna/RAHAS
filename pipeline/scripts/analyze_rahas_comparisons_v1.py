from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ocr.rahas_spatial_proto import build_rahas_spatial_proto_from_checkpoint
from src.ocr.soft_data import build_image_records_from_manifest, build_training_label_maps
from run_rahas_comparison_v1 import (
    CHARACTER_ROOT,
    DATASET_SHA256,
    EXPERIMENTS,
    OUTPUT_ROOT,
    SPLIT_DIR,
    assert_frozen_dataset,
    evaluate,
    manifest_rows,
    parameter_count,
    select_prototype_records,
    sha256,
    write_csv,
)


PROPOSED_CHECKPOINT = Path("pipeline/checkpoints/rahas_source_disjoint_v1/best.pt")
VERIFIED_CHARACTERS = Path("datasets/reference_files/verified_characters.csv")


def safe_label(value: str) -> str:
    return "".join(character if character.isascii() and character.isalnum() else "_" for character in value)[:40]


def write_reference_proposed(project_root: Path, split_dir: Path, output_dir: Path) -> dict:
    frozen = assert_frozen_dataset(split_dir)
    checkpoint = torch.load(PROPOSED_CHECKPOINT, map_location="cpu", weights_only=False)
    if sha256(PROPOSED_CHECKPOINT) != "61290fe62bdad6c4b194f4a188800a376852b446614418ae3c7c62266cd270bd":
        raise RuntimeError("Proposed checkpoint hash does not match the frozen reference")
    maps = build_training_label_maps(CHARACTER_ROOT)
    train_records = build_image_records_from_manifest(split_dir / "train_manifest.csv", maps, project_root, "train")
    val_records = build_image_records_from_manifest(split_dir / "validation_manifest.csv", maps, project_root, "validation")
    test_records = build_image_records_from_manifest(split_dir / "test_manifest.csv", maps, project_root, "test")
    train_rows, _ = manifest_rows(split_dir / "train_manifest.csv", project_root)
    _, val_lookup = manifest_rows(split_dir / "validation_manifest.csv", project_root)
    _, test_lookup = manifest_rows(split_dir / "test_manifest.csv", project_root)
    original_train_counts = Counter(
        int(row["class_index"]) for row in train_rows if row["is_augmented"].lower() == "false"
    )
    args = type(
        "Args",
        (),
        {
            "image_size": int(checkpoint["args"]["image_size"]),
            "workers": int(checkpoint["args"]["workers"]),
            "batch_size": 128,
            "prototype_per_class": int(checkpoint["args"]["prototype_per_class"]),
            "seed": int(checkpoint["args"]["seed"]),
        },
    )()
    model = build_rahas_spatial_proto_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    spec = {
        "model": "rahas_spatial_proto",
        "training": "episodic",
        "representation": "five_channel",
        "description": "Reference RAHAS spatial prototypical recognizer.",
    }
    prototypes = select_prototype_records(train_records, args.prototype_per_class, args.seed)
    val_metrics, val_predictions = evaluate(
        model, val_records, prototypes, spec, args, maps, val_lookup, original_train_counts, device
    )
    test_metrics, test_predictions = evaluate(
        model, test_records, prototypes, spec, args, maps, test_lookup, original_train_counts, device
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "predictions_validation.csv", val_predictions)
    write_csv(output_dir / "predictions_test.csv", test_predictions)
    result = {
        "status": "PASS",
        "experiment_id": "P0_rahas_proposed_reference",
        "model": "RahasSpatialProto",
        "input_representation": "five_channel",
        "description": spec["description"],
        "dataset_sha256": DATASET_SHA256,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": val_metrics,
        "test": test_metrics,
        "training_seconds": None,
        "checkpoint_sha256": sha256(PROPOSED_CHECKPOINT),
        "checkpoint": str(PROPOSED_CHECKPOINT),
        "protocol_deviations": [],
        "frozen_split": frozen,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def base_error_analysis(output_dir: Path, maps: dict, train_rows: list[dict]) -> None:
    with (output_dir / "predictions_test.csv").open(newline="", encoding="utf-8-sig") as handle:
        predictions = list(csv.DictReader(handle))
    base_counts = Counter()
    for row in train_rows:
        if row["is_augmented"].lower() != "false":
            continue
        label = row["class_label"]
        info = next(item for item in maps["records"] if item["label"] == label)
        base_counts[maps["idx_to_base_glyph"][int(info["base_idx"])]] += 1
    labels = maps["idx_to_base_glyph"][1:]
    matrix = {true: Counter() for true in labels}
    pairs = {}
    for row in predictions:
        true_base = row["true_base"]
        predicted_base = row["predicted_base"]
        matrix[true_base][predicted_base] += 1
        if true_base != predicted_base:
            pairs.setdefault((true_base, predicted_base), []).append(row)
    matrix_rows = []
    for true_base in labels:
        matrix_rows.append({"true_base": true_base, **{predicted: matrix[true_base][predicted] for predicted in labels}})
    write_csv(output_dir / "base_confusion_matrix.csv", matrix_rows)
    sheets = output_dir / "base_confusion_sheets"
    sheets.mkdir()
    pair_rows = []
    for rank, ((true_base, predicted_base), rows) in enumerate(
        sorted(pairs.items(), key=lambda item: len(item[1]), reverse=True)[:20], start=1
    ):
        modifier_correct = sum(row["true_modifier"] == row["predicted_modifier"] for row in rows)
        pair_rows.append(
            {
                "rank": rank,
                "true_base": true_base,
                "predicted_base": predicted_base,
                "errors": len(rows),
                "true_base_train_original_samples": base_counts[true_base],
                "predicted_base_train_original_samples": base_counts[predicted_base],
                "modifier_correct_rate": modifier_correct / len(rows),
                "mean_confidence": float(np.mean([float(row["confidence"]) for row in rows])),
                "qualitative_status": "requires expert visual review: candidates may be visually similar, fragmented, or restoration-damaged",
                "sheet": f"base_confusion_sheets/{rank:02d}_{safe_label(true_base)}_to_{safe_label(predicted_base)}.png",
            }
        )
        build_sheet(rows[:12], project_root=Path.cwd(), path=output_dir / pair_rows[-1]["sheet"])
    write_csv(output_dir / "base_confusion_pairs_top20.csv", pair_rows)


def build_sheet(rows: list[dict], project_root: Path, path: Path) -> None:
    columns = 4
    width, height = 144, 158
    canvas = Image.new("RGB", (columns * width, ((len(rows) + columns - 1) // columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        source = project_root / row["sample_path"]
        with Image.open(source) as image:
            image = image.convert("L")
            image.thumbnail((128, 128))
            tile = Image.new("L", (128, 128), 255)
            tile.paste(image, ((128 - image.width) // 2, (128 - image.height) // 2))
            x = (index % columns) * width + 8
            y = (index // columns) * height + 4
            canvas.paste(tile.convert("RGB"), (x, y))
            draw.text((x, y + 132), f"T:{row['true_base']} P:{row['predicted_base']}", fill="black")
    canvas.save(path)


def verified_full_labels(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("Letter_") and value:
                labels.add(value.strip())
    return labels


def build_indexed_sheet(rows: list[dict], project_root: Path, path: Path, pair_rank: int) -> list[dict]:
    columns = 4
    tile_width, tile_height = 144, 154
    canvas = Image.new(
        "RGB",
        (columns * tile_width, ((len(rows) + columns - 1) // columns) * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    index_rows = []
    for index, row in enumerate(rows):
        tile_id = f"P{pair_rank:02d}-S{index + 1:02d}"
        source = project_root / row["sample_path"]
        with Image.open(source) as image:
            image = image.convert("L")
            image.thumbnail((128, 128))
            tile = Image.new("L", (128, 128), 255)
            tile.paste(image, ((128 - image.width) // 2, (128 - image.height) // 2))
        x = (index % columns) * tile_width + 8
        y = (index // columns) * tile_height + 4
        canvas.paste(tile.convert("RGB"), (x, y))
        draw.text((x, y + 132), tile_id, fill="black")
        index_rows.append({"pair_rank": pair_rank, "tile_id": tile_id, **row})
    canvas.save(path)
    return index_rows


def build_confusion_overview(
    ranked_pairs: list[tuple[tuple[str, str], list[dict]]], project_root: Path, path: Path
) -> None:
    samples_per_pair = 6
    tile = 100
    row_height = 112
    canvas = Image.new("RGB", (samples_per_pair * tile, len(ranked_pairs) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for pair_offset, (_, rows) in enumerate(ranked_pairs):
        y = pair_offset * row_height
        draw.text((2, y + 2), f"P{pair_offset + 1:02d}", fill="black")
        for sample_offset, row in enumerate(rows[:samples_per_pair]):
            source = project_root / row["sample_path"]
            with Image.open(source) as image:
                image = image.convert("L")
                image.thumbnail((88, 88))
                image_x = sample_offset * tile + 8
                image_y = y + 18
                cell = Image.new("L", (88, 88), 255)
                cell.paste(image, ((88 - image.width) // 2, (88 - image.height) // 2))
                canvas.paste(cell.convert("RGB"), (image_x, image_y))
    canvas.save(path)


def base_error_analysis_v2(output_dir: Path, maps: dict, train_rows: list[dict]) -> None:
    with (output_dir / "predictions_test.csv").open(newline="", encoding="utf-8-sig") as handle:
        predictions = list(csv.DictReader(handle))
    verified = verified_full_labels(VERIFIED_CHARACTERS)
    base_counts = Counter()
    by_label = {item["label"]: item for item in maps["records"]}
    for row in train_rows:
        if row["is_augmented"].lower() != "false":
            continue
        info = by_label[row["class_label"]]
        base_counts[maps["idx_to_base_glyph"][int(info["base_idx"])]] += 1
    pairs = defaultdict(list)
    for row in predictions:
        if row["true_base"] != row["predicted_base"]:
            pairs[(row["true_base"], row["predicted_base"])].append(row)
    ranked_pairs = sorted(pairs.items(), key=lambda item: len(item[1]), reverse=True)[:20]
    all_sheets = output_dir / "base_confusion_sheets_indexed_v2"
    plain_sheets = output_dir / "base_confusion_sheets_plain_base_v2"
    all_sheets.mkdir(exist_ok=False)
    plain_sheets.mkdir(exist_ok=False)
    index_rows = []
    pair_rows = []
    for rank, ((true_base, predicted_base), rows) in enumerate(ranked_pairs, start=1):
        full_path = all_sheets / f"pair_{rank:02d}.png"
        index_rows.extend(build_indexed_sheet(rows[:12], Path.cwd(), full_path, rank))
        plain_rows = [row for row in rows if row["true_modifier"] == "none"]
        plain_path = plain_sheets / f"pair_{rank:02d}.png"
        if plain_rows:
            build_indexed_sheet(plain_rows[:12], Path.cwd(), plain_path, rank)
        modifier_correct = sum(row["true_modifier"] == row["predicted_modifier"] for row in rows)
        verified_rows = sum(row["class_label"] in verified for row in rows)
        pair_rows.append(
            {
                "rank": rank,
                "true_base": true_base,
                "predicted_base": predicted_base,
                "errors": len(rows),
                "plain_base_errors": len(plain_rows),
                "modifier_bearing_errors": len(rows) - len(plain_rows),
                "verified_full_labels": verified_rows,
                "true_base_train_original_samples": base_counts[true_base],
                "predicted_base_train_original_samples": base_counts[predicted_base],
                "modifier_correct_rate": modifier_correct / len(rows),
                "mean_confidence": float(np.mean([float(row["confidence"]) for row in rows])),
                "full_sheet": full_path.relative_to(output_dir).as_posix(),
                "plain_base_sheet": plain_path.relative_to(output_dir).as_posix() if plain_rows else "",
                "qualitative_review": "pending indexed visual review",
            }
        )
    for row in index_rows:
        row["verified_devanagari_full_label"] = str(row["class_label"] in verified).lower()
    write_csv(output_dir / "base_confusion_pairs_top20_v2.csv", pair_rows)
    write_csv(output_dir / "base_confusion_sheet_index_v2.csv", index_rows)
    build_confusion_overview(ranked_pairs, Path.cwd(), output_dir / "base_confusion_overview_v2.png")


def comparison_table(output_root: Path) -> None:
    result_paths = sorted(output_root.glob("*/metrics.json"))
    rows = []
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        test = result["test"]
        frequency = test.get("frequency_bins", {})
        rows.append(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "model": result["model"],
                "input_representation": result["input_representation"],
                "parameter_count": result["parameter_count"],
                "best_epoch": result["best_epoch"],
                "validation_accuracy": result["validation"]["accuracy"],
                "validation_macro_f1": result["validation"]["macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
                "test_top3": test["top3"],
                "test_top5": test["top5"],
                "base_glyph_accuracy": test["base_accuracy"],
                "modifier_accuracy": test["modifier_accuracy"],
                "auxiliary_metric_source": test["auxiliary_metric_source"],
                "frequency_bins_json": json.dumps(frequency, ensure_ascii=False),
                "training_seconds": result["training_seconds"],
                "inference_ms_per_sample": test["inference_ms_per_sample"],
                "checkpoint_hash": result["checkpoint_sha256"],
            }
        )
    rows.append(
        {
            "experiment_id": "A6_raw_vs_restored",
            "status": "PARTIAL",
            "model": "not_runnable",
            "input_representation": "raw/restored pairs unavailable",
            "parameter_count": None,
            "best_epoch": None,
            "validation_accuracy": None,
            "validation_macro_f1": None,
            "test_accuracy": None,
            "test_balanced_accuracy": None,
            "test_macro_f1": None,
            "test_weighted_f1": None,
            "test_top3": None,
            "test_top5": None,
            "base_glyph_accuracy": None,
            "modifier_accuracy": None,
            "auxiliary_metric_source": "not available",
            "frequency_bins_json": "{}",
            "training_seconds": None,
            "inference_ms_per_sample": None,
            "checkpoint_hash": None,
        }
    )
    write_csv(output_root / "comparison_table.csv", rows)
    markdown = [
        "# RAHAS Source-Disjoint Seed-2026 Comparison",
        "",
        "| Experiment | Model | Test accuracy | Test macro-F1 | Top-3 | Parameters | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "PASS":
            markdown.append(
                f"| {row['experiment_id']} | {row['model']} | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        markdown.append(
            f"| {row['experiment_id']} | {row['model']} | {float(row['test_accuracy']):.4f} | "
            f"{float(row['test_macro_f1']):.4f} | {float(row['test_top3']):.4f} | "
            f"{int(row['parameter_count']):,} | {row['best_epoch']} |"
        )
    (output_root / "comparison_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def write_status(output_root: Path) -> None:
    statuses = {"P0_rahas_proposed_reference": "PASS"}
    statuses.update({name: "PASS" for name in EXPERIMENTS})
    statuses["A6_raw_vs_restored"] = "PARTIAL"
    payload = {
        "dataset_sha256": DATASET_SHA256,
        "experiments": statuses,
        "A6_raw_vs_restored": {
            "status": "PARTIAL",
            "usable_paired_subset_size": 0,
            "reason": "No raw/restored character crop pair field or reliable exact pair could be reconstructed for the frozen manifest samples.",
        },
        "limitations": [
            "All completed results are single-seed (2026) and require repeat-seed confirmation before component claims.",
            "The held-out test represents 76 of 372 classes; it is not a 372-class coverage claim.",
        ],
    }
    (output_root / "experiment_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path.cwd().resolve()
    output_root = OUTPUT_ROOT.resolve()
    split_dir = SPLIT_DIR.resolve()
    reference_dir = output_root / "P0_rahas_proposed_reference"
    if not reference_dir.exists():
        write_reference_proposed(project_root, split_dir, reference_dir)
    if not (reference_dir / "base_confusion_pairs_top20.csv").exists():
        maps = build_training_label_maps(CHARACTER_ROOT)
        train_rows, _ = manifest_rows(split_dir / "train_manifest.csv", project_root)
        base_error_analysis(reference_dir, maps, train_rows)
    if not (reference_dir / "base_confusion_pairs_top20_v2.csv").exists():
        maps = build_training_label_maps(CHARACTER_ROOT)
        train_rows, _ = manifest_rows(split_dir / "train_manifest.csv", project_root)
        base_error_analysis_v2(reference_dir, maps, train_rows)
    comparison_table(output_root)
    write_status(output_root)
    print(f"WROTE comparison artifacts under {output_root}", flush=True)


if __name__ == "__main__":
    main()
