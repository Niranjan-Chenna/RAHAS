from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ocr.rahas_rapt import RahasRAPT, reliability_transport_logits, shot_aware_query_router_logits
from src.ocr.soft_data import (
    SoftFeatureDataset,
    build_image_records_from_manifest,
    build_training_label_maps,
    macro_f1,
)
from train_rahas_rapt_v1 import (
    build_eval_prototypes,
    original_shot_counts,
    select_prototype_records,
    shot_bin_metrics,
    verify_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and evaluate the RAHAS-RAPT shot router.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pipeline/checkpoints/rahas_rapt_v1_seed2026/best.pt"),
    )
    parser.add_argument(
        "--characters",
        type=Path,
        default=Path("datasets/prepared/12_ocr_soft_resized_v1/characters"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("datasets/splits/rahas_source_disjoint_v1"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def collect_logits(
    model,
    records,
    prototype_tokens,
    prototype_reliability,
    prototype_embeddings,
    prototype_labels,
    image_size,
    device,
    workers,
):
    loader = DataLoader(
        SoftFeatureDataset(records, image_size, False, 1),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
    )
    transport = []
    direct = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            query_embeddings = model.tokens_to_embedding(output["tokens"], output["reliability"])
            transport.append(
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
            direct.append(output["direct_full_label"][:, prototype_labels].cpu())
            targets.append(batch["full"])
    return torch.cat(transport), torch.cat(direct), torch.cat(targets)


def metrics_from_logits(logits, target, prototype_labels, shot_counts, num_classes):
    prediction = prototype_labels[logits.argmax(dim=1)]
    top3 = prototype_labels[logits.topk(min(3, logits.shape[1]), dim=1).indices]
    metrics = {
        "accuracy": float((prediction == target).float().mean()),
        "macro_f1": macro_f1(prediction, target, num_classes),
        "top3": float((top3 == target[:, None]).any(dim=1).float().mean()),
    }
    metrics.update(shot_bin_metrics(prediction, target, shot_counts, num_classes))
    return metrics


def main() -> None:
    args = parse_args()
    audit = verify_split(args.split_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    maps = build_training_label_maps(args.characters)
    project_root = Path.cwd()
    train_records = build_image_records_from_manifest(
        args.split_dir / "train_manifest.csv", maps, project_root, "train"
    )
    validation_records = build_image_records_from_manifest(
        args.split_dir / "validation_manifest.csv", maps, project_root, "validation"
    )
    test_records = build_image_records_from_manifest(
        args.split_dir / "test_manifest.csv", maps, project_root, "test"
    )
    shot_counts = original_shot_counts(train_records)
    count_tensor = torch.zeros(checkpoint["model_config"]["num_full_labels"])
    for class_index, count in shot_counts.items():
        count_tensor[class_index] = count

    training_args = checkpoint.get("args", {})
    prototype_records = select_prototype_records(
        train_records,
        int(training_args.get("prototype_per_class", 5)),
        int(training_args.get("seed", 2026)) + 91,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RahasRAPT(**checkpoint["model_config"]).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        "direct_head.weight",
        "direct_head.bias",
        "backbone.imagenet_mean",
        "backbone.imagenet_std",
    }
    if unexpected or not missing.issubset(allowed_missing):
        raise RuntimeError(f"Incompatible checkpoint: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    prototype_tokens, prototype_reliability, prototype_embeddings, prototype_labels = build_eval_prototypes(
        model,
        prototype_records,
        int(training_args.get("image_size", 96)),
        device,
        args.workers,
    )

    val_transport, val_direct, val_targets = collect_logits(
        model,
        validation_records,
        prototype_tokens,
        prototype_reliability,
        prototype_embeddings,
        prototype_labels,
        int(training_args.get("image_size", 96)),
        device,
        args.workers,
    )
    prototype_labels_cpu = prototype_labels.cpu()
    validation_transport_metrics = metrics_from_logits(
        val_transport, val_targets, prototype_labels_cpu, shot_counts, model.config.num_full_labels
    )
    validation_direct_metrics = metrics_from_logits(
        val_direct, val_targets, prototype_labels_cpu, shot_counts, model.config.num_full_labels
    )

    candidates = []
    for max_transport_shots in (1, 2, 4, 9):
        for minimum_margin in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.0):
            routed, use_transport, margins = shot_aware_query_router_logits(
                val_transport,
                val_direct,
                prototype_labels_cpu,
                count_tensor,
                max_transport_shots,
                minimum_margin,
            )
            metrics = metrics_from_logits(
                routed, val_targets, prototype_labels_cpu, shot_counts, model.config.num_full_labels
            )
            candidates.append(
                {
                    "max_transport_shots": max_transport_shots,
                    "minimum_transport_margin": minimum_margin,
                    "validation_metrics": metrics,
                    "validation_transport_fraction": float(use_transport.float().mean()),
                    "validation_mean_transport_margin": float(margins.mean()),
                }
            )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["validation_metrics"]["one_shot_accuracy"]
        >= validation_transport_metrics["one_shot_accuracy"]
    ]
    if not eligible:
        raise RuntimeError("No router candidate preserved transport-only one-shot validation accuracy")
    selected = max(
        eligible,
        key=lambda item: (
            item["validation_metrics"]["macro_f1"],
            item["validation_metrics"]["accuracy"],
        ),
    )

    test_transport, test_direct, test_targets = collect_logits(
        model,
        test_records,
        prototype_tokens,
        prototype_reliability,
        prototype_embeddings,
        prototype_labels,
        int(training_args.get("image_size", 96)),
        device,
        args.workers,
    )
    test_routed, test_use_transport, _ = shot_aware_query_router_logits(
        test_transport,
        test_direct,
        prototype_labels_cpu,
        count_tensor,
        selected["max_transport_shots"],
        selected["minimum_transport_margin"],
    )
    report = {
        "status": "PASS",
        "leakage_audit_status": audit["status"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_rule": (
            "preserve transport-only one-shot validation accuracy; then maximize validation macro_f1 "
            "and validation accuracy"
        ),
        "selected_router": selected,
        "validation": {
            "transport_only": validation_transport_metrics,
            "direct_only": validation_direct_metrics,
        },
        "test": {
            "transport_only": metrics_from_logits(
                test_transport,
                test_targets,
                prototype_labels_cpu,
                shot_counts,
                model.config.num_full_labels,
            ),
            "direct_only": metrics_from_logits(
                test_direct,
                test_targets,
                prototype_labels_cpu,
                shot_counts,
                model.config.num_full_labels,
            ),
            "shot_aware_query_router": metrics_from_logits(
                test_routed,
                test_targets,
                prototype_labels_cpu,
                shot_counts,
                model.config.num_full_labels,
            ),
        },
        "test_transport_fraction": float(test_use_transport.float().mean()),
        "validation_candidates": candidates,
    }
    output = args.output or args.checkpoint.parent / "router_evaluation.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    selected_validation = selected["validation_metrics"]
    routed_test = report["test"]["shot_aware_query_router"]
    print(
        f"selected max_shots={selected['max_transport_shots']} "
        f"margin={selected['minimum_transport_margin']} "
        f"val_acc={selected_validation['accuracy']:.4f} val_macro_f1={selected_validation['macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"TEST routed_acc={routed_test['accuracy']:.4f} routed_macro_f1={routed_test['macro_f1']:.4f} "
        f"one_shot={routed_test['one_shot_accuracy']:.4f} top3={routed_test['top3']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
