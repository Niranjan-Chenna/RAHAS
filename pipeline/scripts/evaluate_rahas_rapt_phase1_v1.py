from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ocr.rahas_rapt import RahasRAPT, reliability_transport_logits, shot_aware_query_router_logits
from src.ocr.soft_data import SoftFeatureDataset, build_image_records_from_manifest, build_training_label_maps
from train_rahas_rapt_v1 import (
    checkpoint_provenance,
    class_map_sha256,
    original_shot_counts,
    select_prototype_records,
    validate_checkpoint_provenance,
    verify_frozen_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the strict Phase-1 RAPT evaluation package.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--characters", type=Path, default=Path("datasets/prepared/12_ocr_soft_resized_v1/characters"))
    parser.add_argument("--split-dir", type=Path, default=Path("datasets/splits/rahas_source_disjoint_v1"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--resnet-validation", type=Path)
    parser.add_argument("--resnet-test", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--prediction-mode",
        choices=("routed", "equal_fusion", "direct_only"),
        default="routed",
    )
    parser.add_argument("--experiment-id")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_test_access(marker: Path, seed: int) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "STARTED", "model": "RAHAS-RAPT", "seed": seed}
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(
            f"RAPT test access was already claimed at {marker}; "
            "restart this seed from a fresh immutable seed directory"
        ) from error


@contextmanager
def immutable_output_directory(output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Immutable evaluation directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        yield staging
        if output.exists():
            raise FileExistsError(f"Immutable evaluation directory was created concurrently: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_path"].replace("\\", "/"): row for row in read_csv(path)}


def build_prototypes(model, records, image_size: int, device, workers: int):
    loader = DataLoader(
        SoftFeatureDataset(records, image_size, False, 0),
        batch_size=96,
        shuffle=False,
        num_workers=workers,
    )
    classes = model.config.num_full_labels
    tokens = torch.zeros(classes, model.config.grid_size**2, model.config.token_dim, device=device)
    reliability = torch.zeros(classes, model.config.grid_size**2, device=device)
    completion = torch.zeros(classes, device=device)
    counts = torch.zeros(classes, device=device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch["full"].to(device)
            output = model(batch["image"].to(device))
            completed, effective, repair_gate = model.complete_support(
                output["tokens"], output["reliability"], batch["base"].to(device), batch["modifier"].to(device)
            )
            tokens.index_add_(0, labels, completed)
            reliability.index_add_(0, labels, effective)
            completion.index_add_(0, labels, repair_gate.mean(1))
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    valid = counts > 0
    prototype_tokens = F.normalize(tokens[valid] / counts[valid, None, None], dim=-1)
    prototype_reliability = (reliability[valid] / counts[valid, None]).clamp(0.0, 1.0)
    prototype_embeddings = model.tokens_to_embedding(prototype_tokens, prototype_reliability)
    prototype_labels = torch.arange(classes, device=device)[valid]
    prototype_completion = completion[valid] / counts[valid]
    return prototype_tokens, prototype_reliability, prototype_embeddings, prototype_labels, prototype_completion


def collect(model, records, prototypes, image_size: int, device, workers: int) -> dict[str, torch.Tensor]:
    prototype_tokens, prototype_reliability, prototype_embeddings, prototype_labels, prototype_completion = prototypes
    loader = DataLoader(
        SoftFeatureDataset(records, image_size, False, 1),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
    )
    result: dict[str, list[torch.Tensor]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            query_embeddings = model.tokens_to_embedding(output["tokens"], output["reliability"])
            result["transport"].append(
                reliability_transport_logits(
                    model,
                    output["tokens"],
                    output["reliability"],
                    query_embeddings,
                    prototype_tokens,
                    prototype_reliability,
                    prototype_embeddings,
                ).cpu()
            )
            result["direct"].append(output["direct_full_label"][:, prototype_labels].cpu())
            result["base"].append(output["base_glyph"].cpu())
            result["modifier"].append(output["modifier"].cpu())
            result["target"].append(batch["full"].cpu())
            result["true_base"].append(batch["base"].cpu())
            result["true_modifier"].append(batch["modifier"].cpu())
            result["query_reliability"].append(output["reliability"].mean(1).cpu())
    bundle = {key: torch.cat(value) for key, value in result.items()}
    bundle["prototype_labels"] = prototype_labels.cpu()
    bundle["prototype_completion"] = prototype_completion.cpu()
    return bundle


def select_router(bundle: dict[str, torch.Tensor], shot_tensor: torch.Tensor) -> tuple[dict, list[dict]]:
    transport = bundle["transport"]
    direct = bundle["direct"]
    target = bundle["target"]
    labels = bundle["prototype_labels"]
    represented_labels = sorted(int(value) for value in target.unique())
    transport_prediction = labels[transport.argmax(1)]
    one_shot = torch.tensor([shot_tensor[int(label)] == 1 for label in target], dtype=torch.bool)
    reference_one_shot = (
        float((transport_prediction[one_shot] == target[one_shot]).float().mean())
        if bool(one_shot.any())
        else 0.0
    )
    candidates = []
    for max_shots in (1, 2, 4, 9):
        for margin in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.0):
            routed, use_transport, _ = shot_aware_query_router_logits(
                transport, direct, labels, shot_tensor, max_shots, margin
            )
            prediction = labels[routed.argmax(1)]
            metrics, _ = classification_metrics(
                prediction.numpy(), target.numpy(), None, None, represented_labels
            )
            one_shot_accuracy = (
                float((prediction[one_shot] == target[one_shot]).float().mean())
                if bool(one_shot.any())
                else 0.0
            )
            candidates.append(
                {
                    "max_transport_shots": max_shots,
                    "minimum_transport_margin": margin,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_one_shot_accuracy": one_shot_accuracy,
                    "validation_transport_fraction": float(use_transport.float().mean()),
                }
            )
    eligible = [row for row in candidates if row["validation_one_shot_accuracy"] >= reference_one_shot]
    if not eligible:
        raise RuntimeError("No validation router preserves transport-only one-shot accuracy")
    selected = max(eligible, key=lambda row: (row["validation_macro_f1"], row["validation_accuracy"]))
    selected["selection_rule"] = "preserve validation transport one-shot accuracy, then maximize validation macro-F1"
    return selected, candidates


def prediction_mode_logits(
    bundle: dict[str, torch.Tensor],
    shot_tensor: torch.Tensor,
    router: dict,
    prediction_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return fixed or validation-selected expert logits without test-time tuning."""

    transport = bundle["transport"]
    direct = bundle["direct"]
    labels = bundle["prototype_labels"]
    transport_log_prob = F.log_softmax(transport, dim=1)
    direct_log_prob = F.log_softmax(direct, dim=1)
    top_values = transport_log_prob.topk(2, dim=1).values
    margins = top_values[:, 0] - top_values[:, 1]
    if prediction_mode == "routed":
        logits, use_transport, margins = shot_aware_query_router_logits(
            transport,
            direct,
            labels,
            shot_tensor,
            router["max_transport_shots"],
            router["minimum_transport_margin"],
        )
        return logits, use_transport, margins, float(use_transport.float().mean())
    if prediction_mode == "equal_fusion":
        use_transport = torch.zeros(len(transport), dtype=torch.bool)
        return 0.5 * (transport_log_prob + direct_log_prob), use_transport, margins, 0.5
    if prediction_mode == "direct_only":
        use_transport = torch.zeros(len(transport), dtype=torch.bool)
        return direct_log_prob, use_transport, margins, 0.0
    raise ValueError(f"unsupported prediction mode: {prediction_mode}")


def classification_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    top_indices: np.ndarray | None,
    true_class_indices: np.ndarray | None,
    represented_labels: list[int] | None = None,
) -> tuple[dict, list[dict]]:
    labels = (
        sorted(set(target.tolist()))
        if represented_labels is None
        else sorted(set(int(label) for label in represented_labels))
    )
    if not set(target.tolist()).issubset(labels):
        raise ValueError("represented_labels must contain every ground-truth class")
    total = len(target)
    rows = []
    for label in labels:
        tp = int(np.sum((target == label) & (prediction == label)))
        fp = int(np.sum((target != label) & (prediction == label)))
        fn = int(np.sum((target == label) & (prediction != label)))
        support = int(np.sum(target == label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_index": label,
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    represented = rows
    weighted = lambda key: sum(row[key] * row["support"] for row in represented) / total if total else 0.0
    result = {
        "samples": total,
        "represented_classes": len(represented),
        "accuracy": float(np.mean(prediction == target)) if total else 0.0,
        "balanced_accuracy": float(np.mean([row["recall"] for row in represented])) if represented else 0.0,
        "macro_precision": float(np.mean([row["precision"] for row in rows])) if rows else 0.0,
        "macro_recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
        "macro_f1": float(np.mean([row["f1"] for row in rows])) if rows else 0.0,
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "weighted_f1": weighted("f1"),
    }
    if top_indices is not None and true_class_indices is not None:
        result["top3"] = float(np.mean(np.any(top_indices[:, :3] == true_class_indices[:, None], axis=1)))
        result["top5"] = float(np.mean(np.any(top_indices[:, :5] == true_class_indices[:, None], axis=1)))
    return result, rows


def frequency_bin(count: int) -> str:
    if count == 0:
        return "zero_shot"
    if count == 1:
        return "one_shot"
    if count <= 4:
        return "2_4"
    if count <= 9:
        return "5_9"
    return "10_plus"


def prediction_package(
    bundle,
    records,
    manifest,
    maps,
    shot_counts,
    shot_tensor,
    router,
    checkpoint_hash,
    split,
    seed,
    dataset_hash,
    prediction_mode="routed",
):
    routed, use_transport, margins, transport_fraction = prediction_mode_logits(
        bundle, shot_tensor, router, prediction_mode
    )
    probability = routed.softmax(1)
    scores, ranked = probability.topk(min(5, probability.shape[1]), dim=1)
    ranked_labels = bundle["prototype_labels"][ranked]
    prediction = ranked_labels[:, 0]
    target = bundle["target"]
    base_prediction = bundle["base"].argmax(1)
    modifier_prediction = bundle["modifier"].argmax(1)
    transport_probability = bundle["transport"].softmax(1)
    transport_best = transport_probability.argmax(1)
    transport_labels = bundle["prototype_labels"][transport_best]
    direct_probability = bundle["direct"].softmax(1)
    full_to_base = torch.zeros(len(maps["idx_to_full_label"]), dtype=torch.long)
    full_to_modifier = torch.zeros(len(maps["idx_to_full_label"]), dtype=torch.long)
    for item in maps["records"]:
        full_to_base[int(item["full_idx"])] = int(item["base_idx"])
        full_to_modifier[int(item["full_idx"])] = int(item["modifier_idx"])
    rows = []
    for index, record in enumerate(records):
        relative = str(record.path.relative_to(Path.cwd())).replace("\\", "/")
        metadata = manifest[relative]
        predicted_index = int(prediction[index])
        true_index = int(target[index])
        row = {
            "sample_path": metadata["sample_path"],
            "sample_id": metadata.get("augmented_sample_id") or metadata.get("original_crop_id") or Path(metadata["sample_path"]).stem,
            "inscription_id": metadata.get("inscription_id", ""),
            "page_id": metadata.get("page_id", ""),
            "word_id": metadata.get("word_id", ""),
            "original_crop_id": metadata.get("original_crop_id", ""),
            "true_full_label": maps["idx_to_full_label"][true_index],
            "true_class_index": true_index,
            "true_base_label": maps["idx_to_base_glyph"][int(bundle["true_base"][index])],
            "true_modifier_label": maps["idx_to_modifier"][int(bundle["true_modifier"][index])],
            "predicted_full_label": maps["idx_to_full_label"][predicted_index],
            "predicted_class_index": predicted_index,
            "predicted_base_label": maps["idx_to_base_glyph"][int(base_prediction[index])],
            "predicted_modifier_label": maps["idx_to_modifier"][int(modifier_prediction[index])],
            "predicted_base_from_full": maps["idx_to_base_glyph"][int(full_to_base[predicted_index])],
            "predicted_modifier_from_full": maps["idx_to_modifier"][int(full_to_modifier[predicted_index])],
            "correct": str(predicted_index == true_index).lower(),
            "confidence": float(scores[index, 0]),
            "selected_route": (
                "transport" if bool(use_transport[index]) else "direct"
            ) if prediction_mode == "routed" else prediction_mode,
            "router_transport_margin": float(margins[index]),
            "router_max_transport_shots": router.get("max_transport_shots", ""),
            "router_minimum_transport_margin": router.get("minimum_transport_margin", ""),
            "completion_score": float(bundle["prototype_completion"][transport_best[index]]),
            "query_reliability": float(bundle["query_reliability"][index]),
            "transport_score": float(transport_probability[index, transport_best[index]]),
            "direct_score": float(direct_probability[index].max()),
            "transport_predicted_label": maps["idx_to_full_label"][int(transport_labels[index])],
            "training_original_sample_count": shot_counts.get(true_index, 0),
            "frequency_bin": frequency_bin(shot_counts.get(true_index, 0)),
            "split": split,
            "model_seed": seed,
            "checkpoint_sha256": checkpoint_hash,
            "dataset_sha256": dataset_hash,
        }
        for rank_index in range(ranked_labels.shape[1]):
            rank = rank_index + 1
            row[f"top_{rank}_label"] = maps["idx_to_full_label"][int(ranked_labels[index, rank_index])]
            row[f"top_{rank}_score"] = float(scores[index, rank_index])
        rows.append(row)
    top_numpy = ranked_labels.numpy()
    represented_labels = sorted(int(value) for value in target.unique())
    metrics, per_class = classification_metrics(
        prediction.numpy(), target.numpy(), top_numpy, target.numpy(), represented_labels
    )
    auxiliary_base_accuracy = float((base_prediction == bundle["true_base"]).float().mean())
    auxiliary_modifier_accuracy = float(
        (modifier_prediction == bundle["true_modifier"]).float().mean()
    )
    full_base_prediction = full_to_base[prediction]
    full_modifier_prediction = full_to_modifier[prediction]
    full_label_base_accuracy = float(
        (full_base_prediction == bundle["true_base"]).float().mean()
    )
    full_label_modifier_accuracy = float(
        (full_modifier_prediction == bundle["true_modifier"]).float().mean()
    )
    metrics.update(
        {
            # Cross-model component metrics are always derived from the final
            # full-label prediction. RAPT head-specific results remain explicit.
            "base_accuracy": full_label_base_accuracy,
            "modifier_accuracy": full_label_modifier_accuracy,
            "auxiliary_base_accuracy": auxiliary_base_accuracy,
            "auxiliary_modifier_accuracy": auxiliary_modifier_accuracy,
            "full_label_base_accuracy": full_label_base_accuracy,
            "full_label_modifier_accuracy": full_label_modifier_accuracy,
            "transport_fraction": transport_fraction,
        }
    )
    counts = torch.tensor([shot_counts.get(int(label), 0) for label in target])
    for name, mask in (
        ("zero_shot", counts == 0),
        ("one_shot", counts == 1),
        ("low_shot_2_4", (counts >= 2) & (counts <= 4)),
        ("mid_shot_5_9", (counts >= 5) & (counts <= 9)),
        ("many_shot_10_plus", counts >= 10),
    ):
        metrics[f"{name}_samples"] = int(mask.sum())
        metrics[f"{name}_classes"] = int(target[mask].unique().numel()) if bool(mask.any()) else 0
        if bool(mask.any()):
            subset_labels = sorted(int(value) for value in target[mask].unique())
            subset_metrics, _ = classification_metrics(
                prediction[mask].numpy(), target[mask].numpy(), None, None, subset_labels
            )
            metrics[f"{name}_accuracy"] = subset_metrics["accuracy"]
            metrics[f"{name}_macro_f1"] = subset_metrics["macro_f1"]
        else:
            metrics[f"{name}_accuracy"] = 0.0
            metrics[f"{name}_macro_f1"] = 0.0
    for item in per_class:
        item["class_label"] = maps["idx_to_full_label"][item["class_index"]]
    return rows, metrics, per_class


def grouped_metrics(rows: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(field) or "UNRESOLVED"].append(row)
    output = []
    for key, items in sorted(groups.items()):
        target = np.asarray([int(row["true_class_index"]) for row in items])
        prediction = np.asarray([int(row["predicted_class_index"]) for row in items])
        metrics, _ = classification_metrics(prediction, target, None, None)
        output.append(
            {
                field: key,
                **metrics,
                "base_accuracy": float(np.mean([row["predicted_base_from_full"] == row["true_base_label"] for row in items])),
                "modifier_accuracy": float(np.mean([row["predicted_modifier_from_full"] == row["true_modifier_label"] for row in items])),
                "auxiliary_base_accuracy": float(np.mean([row["predicted_base_label"] == row["true_base_label"] for row in items])),
                "auxiliary_modifier_accuracy": float(np.mean([row["predicted_modifier_label"] == row["true_modifier_label"] for row in items])),
                "full_label_base_accuracy": float(np.mean([row["predicted_base_from_full"] == row["true_base_label"] for row in items])),
                "full_label_modifier_accuracy": float(np.mean([row["predicted_modifier_from_full"] == row["true_modifier_label"] for row in items])),
            }
        )
    return output


def confusion_outputs(rows: list[dict], maps: dict, output: Path) -> None:
    class_indices = list(range(1, len(maps["idx_to_full_label"])))
    lookup = {value: index for index, value in enumerate(class_indices)}
    matrix = np.zeros((len(class_indices), len(class_indices)), dtype=np.int64)
    for row in rows:
        matrix[lookup[int(row["true_class_index"])], lookup[int(row["predicted_class_index"])]] += 1
    header = ["true_class_index", "true_label"] + [f"pred_{value}" for value in class_indices]
    raw_rows = []
    normalized_rows = []
    for row_index, class_index in enumerate(class_indices):
        raw_rows.append(dict(zip(header, [class_index, maps["idx_to_full_label"][class_index], *matrix[row_index].tolist()])))
        denominator = matrix[row_index].sum()
        normalized = matrix[row_index] / denominator if denominator else matrix[row_index].astype(float)
        normalized_rows.append(dict(zip(header, [class_index, maps["idx_to_full_label"][class_index], *normalized.tolist()])))
    write_csv(output / "confusion_matrix.csv", raw_rows)
    write_csv(output / "confusion_matrix_normalized.csv", normalized_rows)


def make_sheets(rows: list[dict], output: Path, prefix: str, page_size: int = 48) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    tile_w, tile_h, columns = 150, 150, 8
    for page, start in enumerate(range(0, len(rows), page_size), start=1):
        subset = rows[start : start + page_size]
        sheet = Image.new("RGB", (tile_w * columns, tile_h * math.ceil(len(subset) / columns)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(subset):
            number = start + offset + 1
            source = Path(row["sample_path"])
            with Image.open(source) as image:
                tile = ImageOps.contain(image.convert("L"), (118, 110))
            x = (offset % columns) * tile_w
            y = (offset // columns) * tile_h
            sheet.paste(tile.convert("RGB"), (x + (tile_w - tile.width) // 2, y + 22))
            draw.text((x + 4, y + 4), f"{number:04d} T{row['true_class_index']} P{row['predicted_class_index']}", fill="black")
            index_rows.append({"sheet": f"{prefix}_{page:03d}.png", "tile": number, **row})
        sheet.save(output / f"{prefix}_{page:03d}.png")
    return index_rows


def qualitative_outputs(rows: list[dict], output: Path) -> None:
    correct = [row for row in rows if row["correct"] == "true"]
    incorrect = [row for row in rows if row["correct"] == "false"]
    index_rows = make_sheets(correct, output, "correct") + make_sheets(
        incorrect, output, "incorrect"
    )
    write_csv(output / "qualitative_sheet_index.csv", index_rows)


def evaluate_once(args: argparse.Namespace, output: Path) -> None:
    frozen = verify_frozen_split(args.split_dir, include_test=False)
    audit = frozen["audit"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    maps = build_training_label_maps(args.characters)
    provenance = checkpoint_provenance(args.seed, frozen, maps)
    validate_checkpoint_provenance(checkpoint, provenance, maps, "RAPT evaluation checkpoint")
    project_root = Path.cwd()
    train_records = build_image_records_from_manifest(args.split_dir / "train_manifest.csv", maps, project_root, "train")
    validation_records = build_image_records_from_manifest(
        args.split_dir / "validation_manifest.csv", maps, project_root, "validation"
    )
    validation_manifest = manifest_lookup(args.split_dir / "validation_manifest.csv")
    shot_counts = original_shot_counts(train_records)
    shot_tensor = torch.zeros(checkpoint["model_config"]["num_full_labels"])
    for class_index, count in shot_counts.items():
        shot_tensor[class_index] = count
    training_args = checkpoint["args"]
    prototype_records = select_prototype_records(
        train_records, int(training_args["prototype_per_class"]), args.seed + 91
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RahasRAPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    prototypes = build_prototypes(model, prototype_records, int(training_args["image_size"]), device, args.workers)
    validation_bundle = collect(
        model, validation_records, prototypes, int(training_args["image_size"]), device, args.workers
    )
    if args.prediction_mode == "routed":
        router, candidates = select_router(validation_bundle, shot_tensor)
    else:
        router = {
            "prediction_mode": args.prediction_mode,
            "selection_rule": "preregistered fixed inference ablation; no router parameters selected",
        }
        candidates = []
    (output / "router_selection.json").write_text(
        json.dumps({"selected": router, "candidates": candidates}, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint_hash = sha256(args.checkpoint)
    validation_rows, validation_metrics, validation_per_class = prediction_package(
        validation_bundle,
        validation_records,
        validation_manifest,
        maps,
        shot_counts,
        shot_tensor,
        router,
        checkpoint_hash,
        "validation",
        args.seed,
        frozen["dataset_sha256"],
        prediction_mode=args.prediction_mode,
    )
    write_csv(output / "validation_predictions.csv", validation_rows)
    write_csv(output / "validation_per_class_metrics.csv", validation_per_class)

    # Claim access durably before opening test data. A crash after this point is
    # fail-closed: the seed must be restarted rather than evaluating test twice.
    claim_test_access(
        args.output.resolve().parent / f"{args.output.resolve().name}_TEST_ACCESS_STARTED.json",
        args.seed,
    )
    # Full frozen-data verification and the test manifest are deferred until router selection is fixed.
    frozen = verify_frozen_split(args.split_dir, include_test=True)
    test_records = build_image_records_from_manifest(args.split_dir / "test_manifest.csv", maps, project_root, "test")
    test_manifest = manifest_lookup(args.split_dir / "test_manifest.csv")
    test_bundle = collect(model, test_records, prototypes, int(training_args["image_size"]), device, args.workers)
    test_rows, test_metrics, test_per_class = prediction_package(
        test_bundle,
        test_records,
        test_manifest,
        maps,
        shot_counts,
        shot_tensor,
        router,
        checkpoint_hash,
        "test",
        args.seed,
        frozen["dataset_sha256"],
        prediction_mode=args.prediction_mode,
    )
    write_csv(output / "test_predictions.csv", test_rows)
    write_csv(output / "test_per_class_metrics.csv", test_per_class)
    write_csv(output / "frequency_metrics.csv", grouped_metrics(test_rows, "frequency_bin"))
    write_csv(output / "inscription_metrics.csv", grouped_metrics(test_rows, "inscription_id"))
    write_csv(output / "route_metrics.csv", grouped_metrics(test_rows, "selected_route"))
    confusion_outputs(test_rows, maps, output)
    if args.qualitative:
        qualitative_outputs(test_rows, output / "qualitative")
    result = {
        "status": "PASS",
        "protocol": (
            "validation-only checkpoint and router selection; test opened after selection"
            if args.prediction_mode == "routed"
            else "validation-selected checkpoint with preregistered fixed inference mode; test opened after mode was fixed"
        ),
        "prediction_mode": args.prediction_mode,
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "dataset_sha256": frozen["dataset_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "router": router,
        "validation": validation_metrics,
        "test": test_metrics,
        "hashes": {
            "class_map": class_map_sha256(maps),
            "train_manifest": frozen["manifest_sha256"]["train_manifest.csv"],
            "validation_manifest": frozen["manifest_sha256"]["validation_manifest.csv"],
            "test_manifest": frozen["manifest_sha256"]["test_manifest.csv"],
        },
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"PHASE1 seed={args.seed} val_macro_f1={validation_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    with immutable_output_directory(args.output) as output:
        evaluate_once(args, output)


if __name__ == "__main__":
    main()
