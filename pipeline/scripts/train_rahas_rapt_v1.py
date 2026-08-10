from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.rahas_rapt import (
    RahasRAPT,
    aggregate_rapt_prototypes,
    reliability_transport_logits,
    restoration_reliability_target,
)
from src.ocr.soft_data import (
    SoftFeatureDataset,
    build_image_records_from_manifest,
    build_training_label_maps,
    continuous_stroke_attenuation,
    macro_f1,
    soft_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-free RAHAS-RAPT OCR.")
    parser.add_argument("--characters", type=Path, default=Path("datasets/prepared/12_ocr_soft_resized_v1/characters"))
    parser.add_argument("--split-dir", type=Path, default=Path("datasets/splits/rahas_source_disjoint_v1"))
    parser.add_argument("--output", type=Path, default=Path("pipeline/checkpoints/rahas_rapt_v1"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--episodes-per-epoch", type=int, default=30)
    parser.add_argument("--n-way", type=int, default=16)
    parser.add_argument("--support-shots", default="1,2,4")
    parser.add_argument("--q-query", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prototype-per-class", type=int, default=5)
    parser.add_argument("--classification-batches-per-epoch", type=int, default=0)
    parser.add_argument("--classification-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--disable-prototype-completion", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start-rapt", type=Path)
    parser.add_argument("--resnet-classifier-init", type=Path)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after validation-only checkpoint selection; a separate frozen evaluator may open test data later.",
    )
    return parser.parse_args()


class RAPTEpisodicDataset(Dataset):
    def __init__(self, records, image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key):
        index, augmentation_seed, local_class, is_support = key
        record = self.records[index]
        with Image.open(record.path) as image:
            reference = soft_features(
                image.convert("L"),
                self.image_size,
                random.Random(augmentation_seed),
                train=True,
            )
        observed = (
            continuous_stroke_attenuation(reference, random.Random(augmentation_seed + 17))
            if is_support
            else reference
        )
        return {
            "image": observed,
            "reference": reference,
            "full": torch.tensor(record.full_idx),
            "base": torch.tensor(record.base_idx),
            "modifier": torch.tensor(record.modifier_idx),
            "nasal": torch.tensor(record.nasal, dtype=torch.float32),
            "local": torch.tensor(local_class),
            "support": torch.tensor(is_support),
        }


class SourceAwareEpisodeSampler(Sampler[list[tuple[int, int, int, bool]]]):
    """Mixed-shot episodes that avoid augmentation-parent reuse when possible."""

    def __init__(
        self,
        records,
        episodes: int,
        n_way: int,
        support_shots: tuple[int, ...],
        q_query: int,
        seed: int,
    ) -> None:
        self.records = records
        self.by_class = defaultdict(list)
        for index, record in enumerate(records):
            self.by_class[record.full_idx].append(index)
        self.classes = sorted(self.by_class)
        self.episodes = episodes
        self.n_way = min(n_way, len(self.classes))
        self.support_shots = support_shots
        self.q_query = q_query
        self.seed = seed
        self.epoch = 0

    def _parent_key(self, index: int) -> str:
        record = self.records[index]
        return record.augmentation_parent_id or record.original_crop_id or str(record.path)

    def _choose(self, candidates: list[int], needed: int, rng: random.Random) -> list[int]:
        by_parent = defaultdict(list)
        for index in candidates:
            by_parent[self._parent_key(index)].append(index)
        parents = list(by_parent)
        rng.shuffle(parents)
        chosen = [rng.choice(by_parent[parent]) for parent in parents[:needed]]
        if len(chosen) < needed:
            chosen.extend(rng.choices(candidates, k=needed - len(chosen)))
        return chosen

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        self.epoch += 1
        for _ in range(self.episodes):
            selected = rng.sample(self.classes, self.n_way)
            support = []
            query = []
            for local_class, full_class in enumerate(selected):
                k_shot = rng.choice(self.support_shots)
                chosen = self._choose(
                    self.by_class[full_class],
                    k_shot + self.q_query,
                    rng,
                )
                for index in chosen[:k_shot]:
                    support.append((index, rng.randrange(2**31), local_class, True))
                for index in chosen[k_shot:]:
                    query.append((index, rng.randrange(2**31), local_class, False))
            yield support + query

    def __len__(self) -> int:
        return self.episodes


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    similarities = embeddings @ embeddings.transpose(0, 1) / temperature
    identity = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    similarities = similarities.masked_fill(identity, -torch.inf)
    log_probabilities = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    per_anchor = -(log_probabilities.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
    return per_anchor[valid].mean()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_map_sha256(label_maps: dict) -> str:
    payload = {
        key: label_maps[key]
        for key in ("idx_to_full_label", "idx_to_base_glyph", "idx_to_modifier")
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_dataset_hash(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        f"{row['split']}\t{row['sample_path']}\t{row['file_sha256']}"
        for row in sorted(rows, key=lambda item: (item["split"], item["sample_path"]))
        if row["split"] in {"train", "validation", "test"}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_split(split_dir: Path, include_test: bool = True) -> dict:
    with (split_dir / "split_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (split_dir / "leakage_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise RuntimeError(f"Leakage audit is not PASS: {split_dir / 'leakage_audit.json'}")

    expected_hashes = summary.get("manifest_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeError("split_summary.json does not contain frozen manifest hashes")
    deferred = (
        {"canonical_manifest.csv", "test_manifest.csv", "class_distribution.csv"}
        if not include_test
        else set()
    )
    actual_hashes = {}
    for name, expected in expected_hashes.items():
        if name in deferred:
            continue
        actual = sha256(split_dir / name)
        if actual != expected:
            raise RuntimeError(
                f"Frozen manifest hash mismatch for {name}: expected {expected}, got {actual}"
            )
        actual_hashes[name] = actual

    dataset_hash = summary.get("dataset_sha256")
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise RuntimeError("split_summary.json does not contain a valid frozen dataset hash")
    if include_test:
        with (split_dir / "canonical_manifest.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            actual_dataset_hash = _stable_dataset_hash(list(csv.DictReader(handle)))
        if actual_dataset_hash != dataset_hash:
            raise RuntimeError(
                "Frozen dataset hash mismatch: "
                f"expected {dataset_hash}, got {actual_dataset_hash}"
            )
    return {
        "audit": audit,
        "split_seed": int(summary["seed"]),
        "dataset_sha256": dataset_hash,
        "manifest_sha256": dict(expected_hashes),
        "verified_manifest_sha256": actual_hashes,
        "test_material_verified": include_test,
    }


def verify_split(split_dir: Path) -> dict:
    """Backward-compatible full verification used by non-Phase1 callers."""
    return verify_frozen_split(split_dir, include_test=True)["audit"]


def checkpoint_provenance(seed: int, frozen: dict, label_maps: dict) -> dict:
    return {
        "seed": int(seed),
        "dataset_sha256": frozen["dataset_sha256"],
        "manifest_sha256": dict(frozen["manifest_sha256"]),
        "class_map_sha256": class_map_sha256(label_maps),
    }


def validate_checkpoint_provenance(
    checkpoint: dict,
    expected: dict,
    label_maps: dict,
    checkpoint_name: str,
) -> dict:
    stored = checkpoint.get("provenance", {})
    checkpoint_args = checkpoint.get("args", {})
    seed_values = [
        value
        for value in (stored.get("seed"), checkpoint_args.get("seed"))
        if value is not None
    ]
    dataset_hash_values = [
        value
        for value in (stored.get("dataset_sha256"), checkpoint.get("dataset_sha256"))
        if value is not None
    ]
    if not seed_values or any(int(value) != int(expected["seed"]) for value in seed_values):
        raise ValueError(
            f"{checkpoint_name} seed provenance mismatch: "
            f"expected {expected['seed']}, got {seed_values}"
        )
    if not dataset_hash_values or any(
        value != expected["dataset_sha256"] for value in dataset_hash_values
    ):
        raise ValueError(
            f"{checkpoint_name} dataset provenance mismatch: "
            f"expected {expected['dataset_sha256']}, got {dataset_hash_values}"
        )
    stored_maps = checkpoint.get("label_maps")
    if not isinstance(stored_maps, dict):
        raise ValueError(f"{checkpoint_name} does not contain label_maps provenance")
    try:
        actual_class_map_hash = class_map_sha256(stored_maps)
    except (KeyError, TypeError) as error:
        raise ValueError(f"{checkpoint_name} contains incomplete label_maps provenance") from error
    expected_class_map_hash = class_map_sha256(label_maps)
    if actual_class_map_hash != expected_class_map_hash:
        raise ValueError(
            f"{checkpoint_name} class-map provenance mismatch: "
            f"expected {expected_class_map_hash}, got {actual_class_map_hash}"
        )
    declared_class_map_hash = stored.get("class_map_sha256")
    if declared_class_map_hash is not None and declared_class_map_hash != actual_class_map_hash:
        raise ValueError(f"{checkpoint_name} declared class-map hash does not match its label_maps")
    return {
        "seed": int(seed_values[0]),
        "dataset_sha256": dataset_hash_values[0],
        "class_map_sha256": actual_class_map_hash,
    }


def load_resnet_classifier_initialization(
    model: RahasRAPT,
    checkpoint_path: Path,
    label_maps: dict,
    expected_provenance: dict | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = (
        validate_checkpoint_provenance(
            checkpoint, expected_provenance, label_maps, "ResNet initialization checkpoint"
        )
        if expected_provenance is not None
        else {"status": "not_checked_legacy_helper_call"}
    )
    source_maps = checkpoint.get("label_maps", {})
    if source_maps.get("idx_to_full_label") != label_maps.get("idx_to_full_label"):
        raise ValueError("ResNet checkpoint label order does not match the canonical RAHAS label map")
    source = checkpoint["model_state"]
    target = model.state_dict()
    prefixes = {
        "model.conv1.": "backbone.stem.0.",
        "model.bn1.": "backbone.stem.1.",
        "model.layer1.": "backbone.layer1.",
        "model.layer2.": "backbone.layer2.",
        "model.layer3.": "backbone.layer3.",
        "model.layer4.": "backbone.layer4.",
    }
    copied = []
    for source_key, value in source.items():
        for source_prefix, target_prefix in prefixes.items():
            if source_key.startswith(source_prefix):
                target_key = target_prefix + source_key[len(source_prefix) :]
                if target_key not in target or target[target_key].shape != value.shape:
                    raise ValueError(f"Cannot map ResNet parameter {source_key!r} to {target_key!r}")
                target[target_key] = value
                copied.append(target_key)
                break
    model.load_state_dict(target)
    with torch.no_grad():
        classifier_weight = source["model.fc.weight"]
        classifier_bias = source["model.fc.bias"]
        if classifier_weight.shape[0] != model.config.num_full_labels - 1:
            raise ValueError("ResNet classifier does not contain the expected 372 canonical classes")
        model.direct_head.weight.zero_()
        model.direct_head.bias.zero_()
        model.direct_head.weight[1:].copy_(classifier_weight)
        model.direct_head.bias[1:].copy_(classifier_bias)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "provenance": provenance,
        "backbone_tensors_copied": len(copied),
        "classifier_rows_copied": int(classifier_weight.shape[0]),
    }


def select_prototype_records(records, per_class: int, seed: int):
    groups = defaultdict(list)
    for record in records:
        groups[record.full_idx].append(record)
    rng = random.Random(seed)
    selected = []
    for class_records in groups.values():
        originals = [record for record in class_records if not record.is_augmented]
        candidates = originals or class_records
        selected.extend(rng.sample(candidates, min(per_class, len(candidates))))
    return selected


def original_shot_counts(records) -> dict[int, int]:
    counts = defaultdict(int)
    for record in records:
        if not record.is_augmented:
            counts[record.full_idx] += 1
    return dict(counts)


def shot_bin_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shot_counts: dict[int, int],
    num_classes: int,
) -> dict[str, float | int]:
    bins = {
        "zero_shot": lambda count: count == 0,
        "one_shot": lambda count: count == 1,
        "low_shot_2_4": lambda count: 2 <= count <= 4,
        "mid_shot_5_9": lambda count: 5 <= count <= 9,
        "many_shot_10_plus": lambda count: count >= 10,
    }
    result: dict[str, float | int] = {}
    target_shots = torch.tensor([shot_counts.get(int(label), 0) for label in target])
    for name, predicate in bins.items():
        mask = torch.tensor([predicate(int(count)) for count in target_shots], dtype=torch.bool)
        result[f"{name}_samples"] = int(mask.sum())
        result[f"{name}_classes"] = int(target[mask].unique().numel()) if bool(mask.any()) else 0
        result[f"{name}_accuracy"] = (
            float((prediction[mask] == target[mask]).float().mean()) if bool(mask.any()) else 0.0
        )
        result[f"{name}_macro_f1"] = (
            represented_macro_f1(prediction[mask], target[mask]) if bool(mask.any()) else 0.0
        )
    return result


def represented_macro_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    scores = []
    for label in target.unique(sorted=True):
        true_positive = ((prediction == label) & (target == label)).sum().float()
        false_positive = ((prediction == label) & (target != label)).sum().float()
        false_negative = ((prediction != label) & (target == label)).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(float(2 * true_positive / denominator) if denominator > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def build_eval_prototypes(model, records, image_size: int, device, workers: int):
    loader = DataLoader(
        SoftFeatureDataset(records, image_size, False, 0),
        batch_size=96,
        shuffle=False,
        num_workers=workers,
    )
    token_sums = torch.zeros(
        model.config.num_full_labels,
        model.config.grid_size**2,
        model.config.token_dim,
        device=device,
    )
    reliability_sums = torch.zeros(
        model.config.num_full_labels,
        model.config.grid_size**2,
        device=device,
    )
    counts = torch.zeros(model.config.num_full_labels, device=device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch["full"].to(device)
            output = model(batch["image"].to(device))
            completed, effective, _ = model.complete_support(
                output["tokens"],
                output["reliability"],
                batch["base"].to(device),
                batch["modifier"].to(device),
            )
            token_sums.index_add_(0, labels, completed)
            reliability_sums.index_add_(0, labels, effective)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    valid = counts > 0
    tokens = F.normalize(token_sums[valid] / counts[valid, None, None], dim=-1)
    reliability = (reliability_sums[valid] / counts[valid, None]).clamp(0.0, 1.0)
    embeddings = model.tokens_to_embedding(tokens, reliability)
    return tokens, reliability, embeddings, torch.arange(len(counts), device=device)[valid]


def evaluate(
    model,
    prototype_records,
    eval_records,
    image_size: int,
    device,
    workers: int,
    shot_counts: dict[int, int],
):
    prototype_tokens, prototype_reliability, prototype_embeddings, prototype_labels = build_eval_prototypes(
        model,
        prototype_records,
        image_size,
        device,
        workers,
    )
    loader = DataLoader(
        SoftFeatureDataset(eval_records, image_size, False, 1),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
    )
    predictions = []
    targets = []
    top3_correct = base_correct = modifier_correct = seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            labels = batch["full"].to(device)
            query_embeddings = model.tokens_to_embedding(output["tokens"], output["reliability"])
            logits = reliability_transport_logits(
                model,
                output["tokens"],
                output["reliability"],
                query_embeddings,
                prototype_tokens,
                prototype_reliability,
                prototype_embeddings,
            )
            nearest = logits.argmax(dim=1)
            prediction = prototype_labels[nearest]
            top3 = prototype_labels[logits.topk(min(3, logits.shape[1]), dim=1).indices]
            predictions.append(prediction.cpu())
            targets.append(labels.cpu())
            top3_correct += int((top3 == labels[:, None]).any(dim=1).sum())
            base_correct += int((output["base_glyph"].argmax(1) == batch["base"].to(device)).sum())
            modifier_correct += int((output["modifier"].argmax(1) == batch["modifier"].to(device)).sum())
            seen += len(labels)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = {
        "accuracy": float((prediction == target).float().mean()),
        "macro_f1": represented_macro_f1(prediction, target),
        "top3": top3_correct / seen,
        "base_accuracy": base_correct / seen,
        "modifier_accuracy": modifier_correct / seen,
    }
    metrics.update(shot_bin_metrics(prediction, target, shot_counts, model.config.num_full_labels))
    return metrics


def train_classification_epoch(model, loader, optimizer, device, max_batches: int):
    model.train()
    totals = defaultdict(float)
    seen = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        output = model(images)
        direct_loss = F.cross_entropy(output["direct_full_label"], batch["full"].to(device))
        metric_full_loss = F.cross_entropy(output["full_label"], batch["full"].to(device))
        base_loss = F.cross_entropy(output["base_glyph"], batch["base"].to(device))
        modifier_loss = F.cross_entropy(output["modifier"], batch["modifier"].to(device))
        nasal_loss = F.binary_cross_entropy_with_logits(output["nasal"], batch["nasal"].to(device))
        reliability_loss = F.binary_cross_entropy_with_logits(
            output["reliability_logits"],
            restoration_reliability_target(images, model.config.grid_size),
        )
        loss = (
            direct_loss
            + 0.20 * metric_full_loss
            + 0.15 * base_loss
            + 0.08 * modifier_loss
            + 0.03 * nasal_loss
            + 0.10 * reliability_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        count = len(images)
        seen += count
        totals["loss"] += float(loss.detach()) * count
        totals["direct_loss"] += float(direct_loss.detach()) * count
        totals["accuracy"] += int((output["direct_full_label"].argmax(1) == batch["full"].to(device)).sum())
    if seen == 0:
        return {}
    return {key: value / seen for key, value in totals.items()}


def train_epoch(model, loader, optimizer, device):
    model.train()
    totals = defaultdict(float)
    seen_queries = 0
    for batch in loader:
        images = batch["image"].to(device)
        support_mask = batch["support"].to(device).bool()
        local = batch["local"].to(device)
        output = model(images)

        prototype_tokens, prototype_reliability, prototype_embeddings = aggregate_rapt_prototypes(
            model,
            output["tokens"][support_mask],
            output["reliability"][support_mask],
            batch["base"].to(device)[support_mask],
            batch["modifier"].to(device)[support_mask],
            local[support_mask],
        )
        query_embeddings = model.tokens_to_embedding(
            output["tokens"][~support_mask],
            output["reliability"][~support_mask],
        )
        query_logits = reliability_transport_logits(
            model,
            output["tokens"][~support_mask],
            output["reliability"][~support_mask],
            query_embeddings,
            prototype_tokens,
            prototype_reliability,
            prototype_embeddings,
        )
        query_targets = local[~support_mask]
        episodic_loss = F.cross_entropy(query_logits, query_targets)
        full_loss = F.cross_entropy(output["full_label"], batch["full"].to(device))
        direct_loss = F.cross_entropy(output["direct_full_label"], batch["full"].to(device))
        base_loss = F.cross_entropy(output["base_glyph"], batch["base"].to(device))
        base_contrastive = supervised_contrastive_loss(output["embedding"], batch["base"].to(device))
        modifier_loss = F.cross_entropy(output["modifier"], batch["modifier"].to(device))
        nasal_loss = F.binary_cross_entropy_with_logits(output["nasal"], batch["nasal"].to(device))
        reliability_loss = F.binary_cross_entropy_with_logits(
            output["reliability_logits"],
            restoration_reliability_target(images, model.config.grid_size),
        )

        completed, effective, _ = model.complete_support(
            output["tokens"][support_mask],
            output["reliability"][support_mask],
            batch["base"].to(device)[support_mask],
            batch["modifier"].to(device)[support_mask],
        )
        repaired_embedding = model.tokens_to_embedding(completed, effective)
        with torch.no_grad():
            reference_output = model(batch["reference"].to(device)[support_mask])
            reference_embedding = model.tokens_to_embedding(
                reference_output["tokens"],
                reference_output["reliability"],
            )
        repair_loss = (1.0 - F.cosine_similarity(repaired_embedding, reference_embedding)).mean()
        composition_loss = F.cross_entropy(
            model.full_head(repaired_embedding),
            batch["full"].to(device)[support_mask],
        )

        loss = (
            episodic_loss
            + 0.30 * full_loss
            + 0.10 * direct_loss
            + 0.10 * base_loss
            + 0.20 * base_contrastive
            + 0.10 * modifier_loss
            + 0.04 * nasal_loss
            + 0.20 * reliability_loss
            + 0.25 * repair_loss
            + 0.15 * composition_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        count = len(query_targets)
        seen_queries += count
        totals["loss"] += float(loss.detach()) * count
        totals["episodic_loss"] += float(episodic_loss.detach()) * count
        totals["repair_loss"] += float(repair_loss.detach()) * count
        totals["reliability_loss"] += float(reliability_loss.detach()) * count
        totals["accuracy"] += int((query_logits.argmax(1) == query_targets).sum())
    return {key: value / seen_queries for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.resume and (args.warm_start_rapt or args.resnet_classifier_init):
        raise ValueError("--resume cannot be combined with warm-start initialization options")
    support_shots = tuple(sorted({int(value) for value in args.support_shots.split(",") if value.strip()}))
    if not support_shots or min(support_shots) < 1:
        raise ValueError("--support-shots must contain positive comma-separated integers")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frozen = verify_frozen_split(args.split_dir, include_test=False)
    audit = frozen["audit"]
    maps = build_training_label_maps(args.characters)
    provenance = checkpoint_provenance(args.seed, frozen, maps)
    project_root = Path.cwd()
    train_records = build_image_records_from_manifest(
        args.split_dir / "train_manifest.csv", maps, project_root, "train"
    )
    validation_records = build_image_records_from_manifest(
        args.split_dir / "validation_manifest.csv", maps, project_root, "validation"
    )
    prototype_records = select_prototype_records(train_records, args.prototype_per_class, args.seed + 91)
    shot_counts = original_shot_counts(train_records)

    sampler = SourceAwareEpisodeSampler(
        train_records,
        args.episodes_per_epoch,
        args.n_way,
        support_shots,
        args.q_query,
        args.seed,
    )
    loader = DataLoader(
        RAPTEpisodicDataset(train_records, args.image_size),
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    classification_loader = None
    if args.classification_batches_per_epoch > 0:
        classification_loader = DataLoader(
            SoftFeatureDataset(train_records, args.image_size, True, args.seed + 404),
            batch_size=args.classification_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RahasRAPT(
        num_full_labels=len(maps["idx_to_full_label"]),
        num_base_glyphs=len(maps["idx_to_base_glyph"]),
        num_modifiers=len(maps["idx_to_modifier"]),
        token_dim=args.token_dim,
        embedding_dim=args.embedding_dim,
        grid_size=args.grid_size,
        pretrained_backbone=not args.no_pretrained,
        prototype_completion=not args.disable_prototype_completion,
    ).to(device)
    initialization = {}
    if args.warm_start_rapt:
        warm_checkpoint = torch.load(args.warm_start_rapt, map_location=device, weights_only=False)
        warm_provenance = validate_checkpoint_provenance(
            warm_checkpoint, provenance, maps, "warm RAPT checkpoint"
        )
        incompatible = model.load_state_dict(warm_checkpoint["model_state"], strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        allowed_missing = {
            "direct_head.weight",
            "direct_head.bias",
            "backbone.imagenet_mean",
            "backbone.imagenet_std",
        }
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                f"Incompatible warm-start checkpoint: missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        initialization["rapt"] = {
            "checkpoint": str(args.warm_start_rapt),
            "checkpoint_epoch": int(warm_checkpoint.get("epoch", -1)),
            "provenance": warm_provenance,
        }
    if args.resnet_classifier_init:
        initialization["resnet_classifier"] = load_resnet_classifier_initialization(
            model,
            args.resnet_classifier_init,
            maps,
            provenance,
        )
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_scale},
            {"params": other_parameters, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.epochs,
        eta_min=args.lr * 0.05,
    )
    start_epoch = 1
    best_score = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        validate_checkpoint_provenance(checkpoint, provenance, maps, "resume checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        stored_score = checkpoint.get("best_score", best_score)
        best_score = float(stored_score[0] if isinstance(stored_score, (list, tuple)) else stored_score)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "leakage_audit_status": audit["status"],
                "split_dir": str(args.split_dir),
                "provenance": provenance,
                "train_samples": len(train_records),
                "validation_samples": len(validation_records),
                "test_access": "deferred_until_after_checkpoint_selection",
                "prototype_samples": len(prototype_records),
                "support_shots": support_shots,
                "classification_batches_per_epoch": args.classification_batches_per_epoch,
                "device": str(device),
                "model_config": model.checkpoint_config(),
                "initialization": initialization,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    rows = []
    epoch_metrics_path = args.output / "epoch_metrics.csv"
    if args.resume and epoch_metrics_path.is_file():
        with epoch_metrics_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Resume metrics file is empty: {epoch_metrics_path}")
        stored_epochs = [int(row["epoch"]) for row in rows]
        expected_epochs = list(range(1, start_epoch))
        if stored_epochs != expected_epochs:
            raise ValueError(
                f"Resume epoch history mismatch: expected {expected_epochs}, got {stored_epochs}"
            )
    print(
        f"leakage={audit['status']} train={len(train_records)} val={len(validation_records)} "
        f"test=deferred classes={len(set(record.full_idx for record in train_records))} "
        f"shots={support_shots} device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        classification_metrics = (
            train_classification_epoch(
                model,
                classification_loader,
                optimizer,
                device,
                args.classification_batches_per_epoch,
            )
            if classification_loader is not None
            else {}
        )
        train_metrics = train_epoch(model, loader, optimizer, device)
        train_seconds = time.perf_counter() - started
        validation_metrics = evaluate(
            model,
            prototype_records,
            validation_records,
            args.image_size,
            device,
            args.workers,
            shot_counts,
        )
        scheduler.step()
        score = float(validation_metrics["macro_f1"])
        improved = score > best_score
        if improved:
            best_score = score
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": model.checkpoint_config(),
            "label_maps": maps,
            "args": vars(args),
            "dataset_sha256": provenance["dataset_sha256"],
            "provenance": provenance,
            "train_metrics": train_metrics,
            "classification_metrics": classification_metrics,
            "validation_metrics": validation_metrics,
            "best_score": best_score,
        }
        torch.save(state, args.output / "latest.pt")
        if improved:
            torch.save(state, args.output / "best.pt")
        row = {
            "epoch": epoch,
            **{f"classification_{key}": value for key, value in classification_metrics.items()},
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "lr": optimizer.param_groups[1]["lr"],
            "seconds": train_seconds,
            "best": improved,
        }
        rows.append(row)
        with epoch_metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"EPOCH {epoch}/{args.epochs} seconds={train_seconds:.1f} "
            f"class_acc={classification_metrics.get('accuracy', 0.0):.4f} "
            f"train_loss={train_metrics['loss']:.4f} episode_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} macro_f1={validation_metrics['macro_f1']:.4f} "
            f"one_shot={validation_metrics['one_shot_accuracy']:.4f} "
            f"top3={validation_metrics['top3']:.4f} base={validation_metrics['base_accuracy']:.4f} "
            f"modifier={validation_metrics['modifier_accuracy']:.4f} best={'yes' if improved else 'no'}",
            flush=True,
        )

    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    selected_epoch = int(best["epoch"])
    matching_rows = [row for row in rows if int(row["epoch"]) == selected_epoch]
    if len(matching_rows) != 1:
        raise ValueError(f"Expected exactly one epoch_metrics row for selected epoch {selected_epoch}")
    for row in rows:
        row["best"] = int(row["epoch"]) == selected_epoch
    with epoch_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frozen = verify_frozen_split(args.split_dir, include_test=True)
    if args.skip_test:
        with (args.output / "selection_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selected_checkpoint_epoch": int(best["epoch"]),
                    "best_validation_macro_f1": float(best["best_score"]),
                    "test_access": "not_accessed",
                    "split_dir": str(args.split_dir),
                    "leakage_audit_status": audit["status"],
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(
            f"SELECTED epoch={best['epoch']} validation_macro_f1={float(best['best_score']):.4f} "
            "test=not_accessed",
            flush=True,
        )
        return
    test_records = build_image_records_from_manifest(
        args.split_dir / "test_manifest.csv", maps, project_root, "test"
    )
    test_metrics = evaluate(
        model,
        prototype_records,
        test_records,
        args.image_size,
        device,
        args.workers,
        shot_counts,
    )
    with (args.output / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_checkpoint_epoch": int(best["epoch"]),
                "best_validation_macro_f1": float(best["best_score"]),
                "test_metrics": test_metrics,
                "split_dir": str(args.split_dir),
                "leakage_audit_status": audit["status"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(
        f"TEST accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f} "
        f"top3={test_metrics['top3']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
