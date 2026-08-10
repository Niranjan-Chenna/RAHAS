from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.comparison_models import (
    ComparisonProtoNet,
    PlainProtoNet,
    SimpleCNNClassifier,
    TorchvisionClassifier,
    parameter_count,
)
from src.ocr.rahas_spatial_proto import squared_euclidean_logits
from src.ocr.soft_data import (
    OCRImageRecord,
    augment_grayscale,
    build_image_records_from_manifest,
    build_training_label_maps,
    fit_crop_onto_canvas,
    pil_to_gray_array,
    soft_features,
)


DATASET_SHA256 = "4d241e39f754b8cb4271eb94194eb07a706d50ccf61cf966063e87f91b0a8d7b"
SPLIT_DIR = Path("datasets/splits/rahas_source_disjoint_v1")
CHARACTER_ROOT = Path("datasets/prepared/12_ocr_soft_resized_v1/characters")
OUTPUT_ROOT = Path("pipeline/experiments/rahas_source_disjoint_v1_comparisons")


EXPERIMENTS = {
    "B1_simple_cnn": {
        "model": "simple_cnn",
        "training": "supervised",
        "representation": "grayscale",
        "description": "Simple supervised CNN, 372-way cross-entropy.",
    },
    "B2_resnet18_random": {
        "model": "resnet18",
        "training": "supervised",
        "representation": "rgb_grayscale",
        "pretrained": False,
        "description": "Randomly initialized ResNet-18, 372-way cross-entropy.",
    },
    "B2_resnet18_pretrained": {
        "model": "resnet18",
        "training": "supervised",
        "representation": "rgb_grayscale_imagenet",
        "pretrained": True,
        "description": "ImageNet-pretrained ResNet-18, 372-way cross-entropy.",
    },
    "B3_efficientnet_b0_pretrained": {
        "model": "efficientnet_b0",
        "training": "supervised",
        "representation": "rgb_grayscale_imagenet",
        "pretrained": True,
        "description": "ImageNet-pretrained EfficientNet-B0, 372-way cross-entropy.",
    },
    "B4_standard_proto": {
        "model": "plain_proto",
        "training": "episodic",
        "representation": "grayscale",
        "loss": "proto_only",
        "description": "Standard global-pooled prototypical network with ordinary grayscale input.",
    },
    "A1_no_coordinates": {
        "model": "comparison_proto",
        "training": "episodic",
        "representation": "five_channel",
        "use_coordinates": False,
        "preserve_spatial_grid": True,
        "loss": "proposed",
        "description": "RAHAS model without x/y coordinate channels.",
    },
    "A2_global_average_pool": {
        "model": "comparison_proto",
        "training": "episodic",
        "representation": "five_channel",
        "use_coordinates": True,
        "preserve_spatial_grid": False,
        "loss": "proposed",
        "description": "RAHAS model with global average pooling instead of the spatial grid.",
    },
    "A3_proto_loss_only": {
        "model": "comparison_proto",
        "training": "episodic",
        "representation": "five_channel",
        "use_coordinates": True,
        "preserve_spatial_grid": True,
        "loss": "proto_only",
        "description": "RAHAS architecture trained only with character-level prototypical loss.",
    },
    "A4_single_channel": {
        "model": "comparison_proto",
        "training": "episodic",
        "representation": "grayscale",
        "use_coordinates": True,
        "preserve_spatial_grid": True,
        "loss": "proposed",
        "description": "RAHAS architecture using one grayscale channel instead of five-channel evidence.",
    },
    "A5_no_augmentation": {
        "model": "comparison_proto",
        "training": "episodic",
        "representation": "five_channel",
        "use_coordinates": True,
        "preserve_spatial_grid": True,
        "loss": "proposed",
        "original_train_only": True,
        "online_augmentation": False,
        "description": "RAHAS model trained on 2,398 originals with saved and online augmentation disabled.",
    },
    "A7_natural_class_sampling": {
        "model": "comparison_proto",
        "training": "episodic",
        "representation": "five_channel",
        "use_coordinates": True,
        "preserve_spatial_grid": True,
        "loss": "proposed",
        "class_sampling": "natural",
        "description": "RAHAS episodic training with class selection weighted by natural train frequency.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one immutable RAHAS baseline or ablation.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--characters", type=Path, default=CHARACTER_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--episodes-per-epoch", type=int, default=20)
    parser.add_argument("--n-way", type=int, default=16)
    parser.add_argument("--k-shot", type=int, default=2)
    parser.add_argument("--q-query", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--prototype-per-class", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Stop after validation-only checkpoint selection; evaluate the frozen checkpoint separately.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_dataset(split_dir: Path, include_test: bool = True) -> dict:
    summary = json.loads((split_dir / "split_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((split_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    if summary["dataset_sha256"] != DATASET_SHA256:
        raise RuntimeError(f"Dataset hash mismatch: {summary['dataset_sha256']}")
    if audit["status"] != "PASS" or not all(audit["checks"].values()):
        raise RuntimeError("Frozen split leakage audit is not PASS")
    for name, expected in summary["manifest_sha256"].items():
        if not include_test and name in {"canonical_manifest.csv", "test_manifest.csv"}:
            continue
        actual = sha256(split_dir / name)
        if actual != expected:
            raise RuntimeError(f"Manifest hash mismatch for {name}: {actual}")
    return {
        "dataset_sha256": summary["dataset_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "leakage_status": audit["status"],
    }


def manifest_rows(path: Path, project_root: Path) -> tuple[list[dict], dict[Path, dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_path = {}
    for row in rows:
        sample_path = Path(row["sample_path"])
        resolved = (sample_path if sample_path.is_absolute() else project_root / sample_path).resolve()
        by_path[resolved] = row
    return rows, by_path


def ordinary_features(
    image: Image.Image,
    size: int,
    rng: random.Random | None,
    train: bool,
    channels: int,
    imagenet_normalize: bool,
) -> torch.Tensor:
    if train and rng is not None:
        image = augment_grayscale(image, rng)
    gray = pil_to_gray_array(fit_crop_onto_canvas(image, size, rng, train))
    tensor = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0)
    if channels == 3:
        tensor = tensor.expand(3, -1, -1).clone()
        if imagenet_normalize:
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            tensor = (tensor - mean) / std
    return tensor


def representation_features(
    image: Image.Image,
    representation: str,
    size: int,
    rng: random.Random | None,
    train: bool,
) -> torch.Tensor:
    if representation == "five_channel":
        return soft_features(image, size, rng, train)
    if representation == "grayscale":
        return ordinary_features(image, size, rng, train, 1, False)
    if representation == "rgb_grayscale":
        return ordinary_features(image, size, rng, train, 3, False)
    if representation == "rgb_grayscale_imagenet":
        return ordinary_features(image, size, rng, train, 3, True)
    raise ValueError(f"Unknown representation: {representation}")


class RecordDataset(Dataset):
    def __init__(
        self,
        records: list[OCRImageRecord],
        image_size: int,
        representation: str,
        train: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.representation = representation
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        rng = random.Random(self.seed + index * 1009) if self.train else None
        with Image.open(record.path) as image:
            features = representation_features(
                image.convert("L"), self.representation, self.image_size, rng, self.train
            )
        return {
            "image": features,
            "full": torch.tensor(record.full_idx),
            "supervised_full": torch.tensor(record.full_idx - 1),
            "base": torch.tensor(record.base_idx),
            "modifier": torch.tensor(record.modifier_idx),
            "nasal": torch.tensor(record.nasal, dtype=torch.float32),
        }


class EpisodicDataset(Dataset):
    def __init__(
        self,
        records: list[OCRImageRecord],
        image_size: int,
        representation: str,
        online_augmentation: bool,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.representation = representation
        self.online_augmentation = online_augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key) -> dict[str, torch.Tensor]:
        index, augmentation_seed, local_class, is_support = key
        record = self.records[index]
        rng = random.Random(augmentation_seed) if self.online_augmentation else None
        with Image.open(record.path) as image:
            features = representation_features(
                image.convert("L"), self.representation, self.image_size, rng, self.online_augmentation
            )
        return {
            "image": features,
            "full": torch.tensor(record.full_idx),
            "base": torch.tensor(record.base_idx),
            "modifier": torch.tensor(record.modifier_idx),
            "nasal": torch.tensor(record.nasal, dtype=torch.float32),
            "local": torch.tensor(local_class),
            "support": torch.tensor(is_support),
        }


class EpisodeSampler(Sampler[list[tuple[int, int, int, bool]]]):
    def __init__(
        self,
        records: list[OCRImageRecord],
        episodes: int,
        n_way: int,
        k_shot: int,
        q_query: int,
        seed: int,
        class_sampling: str,
    ) -> None:
        self.by_class = defaultdict(list)
        for index, record in enumerate(records):
            self.by_class[record.full_idx].append(index)
        self.classes = sorted(self.by_class)
        self.weights = [len(self.by_class[class_index]) for class_index in self.classes]
        self.episodes = episodes
        self.n_way = min(n_way, len(self.classes))
        self.k_shot = k_shot
        self.q_query = q_query
        self.seed = seed
        self.class_sampling = class_sampling
        self.epoch = 0

    def _classes(self, rng: random.Random) -> list[int]:
        if self.class_sampling == "balanced":
            return rng.sample(self.classes, self.n_way)
        remaining = list(self.classes)
        remaining_weights = list(self.weights)
        selected = []
        for _ in range(self.n_way):
            choice = rng.choices(range(len(remaining)), weights=remaining_weights, k=1)[0]
            selected.append(remaining.pop(choice))
            remaining_weights.pop(choice)
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        self.epoch += 1
        for _ in range(self.episodes):
            selected = self._classes(rng)
            support = []
            query = []
            for local_class, full_class in enumerate(selected):
                candidates = self.by_class[full_class]
                needed = self.k_shot + self.q_query
                chosen = rng.sample(candidates, needed) if len(candidates) >= needed else rng.choices(candidates, k=needed)
                for index in chosen[: self.k_shot]:
                    support.append((index, rng.randrange(2**31), local_class, True))
                for index in chosen[self.k_shot :]:
                    query.append((index, rng.randrange(2**31), local_class, False))
            yield support + query

    def __len__(self) -> int:
        return self.episodes


def build_model(spec: dict, maps: dict, args: argparse.Namespace) -> torch.nn.Module:
    model_name = spec["model"]
    if model_name == "simple_cnn":
        return SimpleCNNClassifier(372)
    if model_name in {"resnet18", "efficientnet_b0"}:
        return TorchvisionClassifier(model_name, bool(spec.get("pretrained", False)), 372)
    if model_name == "plain_proto":
        return PlainProtoNet(1, args.base_channels, args.embedding_dim)
    if model_name == "comparison_proto":
        in_channels = 5 if spec["representation"] == "five_channel" else 1
        return ComparisonProtoNet(
            num_full_labels=len(maps["idx_to_full_label"]),
            num_base_glyphs=len(maps["idx_to_base_glyph"]),
            num_modifiers=len(maps["idx_to_modifier"]),
            in_channels=in_channels,
            base_channels=args.base_channels,
            embedding_dim=args.embedding_dim,
            grid_size=args.grid_size,
            use_coordinates=bool(spec.get("use_coordinates", True)),
            preserve_spatial_grid=bool(spec.get("preserve_spatial_grid", True)),
            auxiliary_heads=spec.get("loss") != "proto_only",
        )
    raise ValueError(f"Unknown model: {model_name}")


def select_prototype_records(
    records: list[OCRImageRecord], per_class: int, seed: int
) -> list[OCRImageRecord]:
    groups = defaultdict(list)
    for record in records:
        groups[record.full_idx].append(record)
    selected = []
    rng = random.Random(seed + 91)
    for class_records in groups.values():
        selected.extend(rng.sample(class_records, min(per_class, len(class_records))))
    return selected


def build_prototypes(
    model: torch.nn.Module,
    records: list[OCRImageRecord],
    representation: str,
    image_size: int,
    device: torch.device,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        RecordDataset(records, image_size, representation, False, 0),
        batch_size=128,
        shuffle=False,
        num_workers=workers,
    )
    sums = torch.zeros(373, 128, device=device)
    counts = torch.zeros(373, device=device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch["full"].to(device)
            embeddings = model(batch["image"].to(device))["embedding"]
            if sums.shape[1] != embeddings.shape[1]:
                sums = torch.zeros(373, embeddings.shape[1], device=device)
            sums.index_add_(0, labels, embeddings)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    valid = counts > 0
    prototypes = F.normalize(sums[valid] / counts[valid, None], dim=1)
    return prototypes, torch.arange(len(counts), device=device)[valid]


def class_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    # Keep the macro class universe fixed to classes represented by ground truth.
    # Prediction-only classes still contribute false positives to their target
    # classes, but cannot change the denominator used for checkpoint selection.
    labels = sorted(set(target.tolist()))
    f1_values = []
    recalls = []
    weighted_f1 = 0.0
    total = len(target)
    for label in labels:
        tp = int(np.sum((prediction == label) & (target == label)))
        fp = int(np.sum((prediction == label) & (target != label)))
        fn = int(np.sum((prediction != label) & (target == label)))
        support = int(np.sum(target == label))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        if denominator:
            f1_values.append(f1)
        if support:
            recalls.append(tp / support)
            weighted_f1 += f1 * support
    return {
        "accuracy": float(np.mean(prediction == target)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "weighted_f1": weighted_f1 / total if total else 0.0,
    }


def frequency_bin(count: int) -> str:
    if count <= 1:
        return "one_shot"
    if count <= 4:
        return "2_4"
    if count <= 9:
        return "5_9"
    if count <= 19:
        return "10_19"
    return "20_plus"


def evaluate(
    model: torch.nn.Module,
    records: list[OCRImageRecord],
    prototype_records: list[OCRImageRecord] | None,
    spec: dict,
    args: argparse.Namespace,
    maps: dict,
    row_lookup: dict[Path, dict],
    original_train_counts: Counter,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    prototypes = prototype_labels = None
    if spec["training"] == "episodic":
        prototypes, prototype_labels = build_prototypes(
            model, prototype_records or [], spec["representation"], args.image_size, device, args.workers
        )
    loader = DataLoader(
        RecordDataset(records, args.image_size, spec["representation"], False, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    predictions = []
    targets = []
    top_indices = []
    top_scores = []
    confidences = []
    direct_bases = []
    direct_modifiers = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            if spec["training"] == "episodic":
                logits = squared_euclidean_logits(output["embedding"], prototypes)
                ranked = logits.topk(min(5, logits.shape[1]), dim=1).indices
                full_ranked = prototype_labels[ranked]
                prediction = full_ranked[:, 0]
            else:
                logits = output["full_label"]
                ranked = logits.topk(min(5, logits.shape[1]), dim=1).indices
                full_ranked = ranked + 1
                prediction = full_ranked[:, 0]
            probability = logits.softmax(1)
            confidences.extend(probability.gather(1, ranked[:, :1]).squeeze(1).cpu().tolist())
            top_scores.extend(probability.gather(1, ranked).cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            targets.extend(batch["full"].tolist())
            top_indices.extend(full_ranked.cpu().tolist())
            if "base_glyph" in output:
                direct_bases.extend(output["base_glyph"].argmax(1).cpu().tolist())
                direct_modifiers.extend(output["modifier"].argmax(1).cpu().tolist())
    elapsed = time.perf_counter() - started
    prediction = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    full_to_base = np.zeros(len(maps["idx_to_full_label"]), dtype=np.int64)
    full_to_modifier = np.zeros(len(maps["idx_to_full_label"]), dtype=np.int64)
    for record in maps["records"]:
        full_to_base[int(record["full_idx"])] = int(record["base_idx"])
        full_to_modifier[int(record["full_idx"])] = int(record["modifier_idx"])
    true_base = full_to_base[target]
    true_modifier = full_to_modifier[target]
    derived_base = full_to_base[prediction]
    derived_modifier = full_to_modifier[prediction]
    has_direct = len(direct_bases) == len(records)
    chosen_base = np.asarray(direct_bases, dtype=np.int64) if has_direct else derived_base
    chosen_modifier = np.asarray(direct_modifiers, dtype=np.int64) if has_direct else derived_modifier
    metrics = class_metrics(prediction, target)
    ranked_array = np.asarray(top_indices, dtype=np.int64)
    metrics.update(
        {
            "top3": float(np.mean(np.any(ranked_array[:, :3] == target[:, None], axis=1))),
            "top5": float(np.mean(np.any(ranked_array[:, :5] == target[:, None], axis=1))),
            "base_accuracy": float(np.mean(chosen_base == true_base)),
            "modifier_accuracy": float(np.mean(chosen_modifier == true_modifier)),
            "base_accuracy_from_character": float(np.mean(derived_base == true_base)),
            "modifier_accuracy_from_character": float(np.mean(derived_modifier == true_modifier)),
            "auxiliary_metric_source": "direct_heads" if has_direct else "derived_from_character_prediction",
            "inference_seconds": elapsed,
            "inference_ms_per_sample": elapsed * 1000.0 / len(records),
        }
    )
    conditions = Counter()
    frequency_results = {}
    for index in range(len(records)):
        base_ok = chosen_base[index] == true_base[index]
        modifier_ok = chosen_modifier[index] == true_modifier[index]
        conditions[f"base_{'correct' if base_ok else 'incorrect'}__modifier_{'correct' if modifier_ok else 'incorrect'}"] += 1
    metrics["base_modifier_conditions"] = dict(conditions)
    bins = [frequency_bin(original_train_counts[int(value)]) for value in target]
    for name in ["one_shot", "2_4", "5_9", "10_19", "20_plus"]:
        mask = np.asarray([value == name for value in bins])
        frequency_results[name] = {
            "samples": int(mask.sum()),
            "character_accuracy": float(np.mean(prediction[mask] == target[mask])) if mask.any() else None,
            "base_accuracy": float(np.mean(chosen_base[mask] == true_base[mask])) if mask.any() else None,
            "modifier_accuracy": float(np.mean(chosen_modifier[mask] == true_modifier[mask])) if mask.any() else None,
        }
    metrics["frequency_bins"] = frequency_results
    rows = []
    for index, record in enumerate(records):
        metadata = row_lookup[record.path.resolve()]
        top_labels = [maps["idx_to_full_label"][value] for value in top_indices[index]]
        row = {
                "sample_path": metadata["sample_path"],
                "sample_id": metadata.get("augmented_sample_id") or metadata.get("original_crop_id") or Path(metadata["sample_path"]).stem,
                "split": metadata["split"],
                "class_label": record.label,
                "class_index": record.full_idx,
                "predicted_label": maps["idx_to_full_label"][prediction[index]],
                "predicted_class_index": int(prediction[index]),
                "correct": str(bool(prediction[index] == target[index])).lower(),
                "confidence": confidences[index],
                "top5_labels": "|".join(top_labels),
                "true_base": maps["idx_to_base_glyph"][true_base[index]],
                "predicted_base": maps["idx_to_base_glyph"][chosen_base[index]],
                "predicted_base_from_character": maps["idx_to_base_glyph"][derived_base[index]],
                "true_modifier": maps["idx_to_modifier"][true_modifier[index]],
                "predicted_modifier": maps["idx_to_modifier"][chosen_modifier[index]],
                "predicted_modifier_from_character": maps["idx_to_modifier"][derived_modifier[index]],
                "training_original_count": original_train_counts[record.full_idx],
                "frequency_bin": bins[index],
                "inscription_id": metadata.get("inscription_id", ""),
                "page_id": metadata.get("page_id", ""),
                "word_id": metadata.get("word_id", ""),
                "original_crop_id": metadata.get("original_crop_id", ""),
            }
        for rank, (label, score) in enumerate(zip(top_labels, top_scores[index]), start=1):
            row[f"top_{rank}_label"] = label
            row[f"top_{rank}_score"] = float(score)
        rows.append(row)
    return metrics, rows


def train_supervised_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    total_loss = total_correct = total = 0
    for batch in loader:
        output = model(batch["image"].to(device))
        targets = batch["supervised_full"].to(device)
        loss = F.cross_entropy(output["full_label"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss) * len(targets)
        total_correct += int((output["full_label"].argmax(1) == targets).sum())
        total += len(targets)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def train_episodic_epoch(model, loader, optimizer, device, loss_mode: str) -> dict[str, float]:
    model.train()
    totals = defaultdict(float)
    seen_queries = 0
    for batch in loader:
        output = model(batch["image"].to(device))
        local = batch["local"].to(device)
        support_mask = batch["support"].to(device).bool()
        local_classes = int(local.max()) + 1
        prototypes = torch.stack(
            [output["embedding"][support_mask & (local == index)].mean(0) for index in range(local_classes)]
        )
        prototypes = F.normalize(prototypes, dim=1)
        query_logits = squared_euclidean_logits(output["embedding"][~support_mask], prototypes)
        query_targets = local[~support_mask]
        proto_loss = F.cross_entropy(query_logits, query_targets)
        loss = proto_loss
        if loss_mode == "proposed":
            full_loss = F.cross_entropy(output["full_label"], batch["full"].to(device))
            base_loss = F.cross_entropy(output["base_glyph"], batch["base"].to(device))
            modifier_loss = F.cross_entropy(output["modifier"], batch["modifier"].to(device))
            nasal_loss = F.binary_cross_entropy_with_logits(output["nasal"], batch["nasal"].to(device))
            loss = proto_loss + 0.30 * full_loss + 0.30 * base_loss + 0.15 * modifier_loss + 0.05 * nasal_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        count = len(query_targets)
        seen_queries += count
        totals["loss"] += float(loss) * count
        totals["proto_loss"] += float(proto_loss) * count
        totals["accuracy"] += int((query_logits.argmax(1) == query_targets).sum())
    return {key: value / seen_queries for key, value in totals.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mark_best_epoch(rows: list[dict], selected_epoch: int) -> None:
    matching_rows = [row for row in rows if int(row["epoch"]) == selected_epoch]
    if len(matching_rows) != 1:
        raise ValueError(f"Expected exactly one epoch_metrics row for selected epoch {selected_epoch}")
    for row in rows:
        row["best"] = row is matching_rows[0]


def main() -> None:
    args = parse_args()
    spec = dict(EXPERIMENTS[args.experiment])
    project_root = Path.cwd().resolve()
    split_dir = args.split_dir.resolve()
    characters = args.characters.resolve()
    output_dir = (args.output_root / args.experiment).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Immutable experiment directory already exists: {output_dir}")
    frozen = assert_frozen_dataset(split_dir, include_test=not args.skip_test)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maps = build_training_label_maps(characters)
    train_records = build_image_records_from_manifest(split_dir / "train_manifest.csv", maps, project_root, "train")
    val_records = build_image_records_from_manifest(split_dir / "validation_manifest.csv", maps, project_root, "validation")
    train_rows, train_lookup = manifest_rows(split_dir / "train_manifest.csv", project_root)
    _, val_lookup = manifest_rows(split_dir / "validation_manifest.csv", project_root)
    original_train_counts = Counter(
        int(row["class_index"]) for row in train_rows if row["is_augmented"].lower() == "false"
    )
    if spec.get("original_train_only"):
        original_paths = {
            path for path, row in train_lookup.items() if row["is_augmented"].lower() == "false"
        }
        train_records = [record for record in train_records if record.path.resolve() in original_paths]
    if len(set(record.full_idx for record in train_records)) != 372:
        raise RuntimeError("Training records do not cover all 372 classes")
    model = build_model(spec, maps, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    output_dir.mkdir(parents=True)
    preflight = {
        **frozen,
        "experiment": args.experiment,
        "seed": args.seed,
        "samples": {"train": len(train_records), "validation": len(val_records)},
        "test_access": "deferred_until_after_checkpoint_selection",
        "classes": {
            "train": len(set(record.full_idx for record in train_records)),
            "validation": len(set(record.full_idx for record in val_records)),
        },
        "parameter_count": parameter_count(model),
        "device": str(device),
        "spec": spec,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prototype_records = None
    if spec["training"] == "episodic":
        prototype_records = select_prototype_records(train_records, args.prototype_per_class, args.seed)
        sampler = EpisodeSampler(
            train_records,
            args.episodes_per_epoch,
            args.n_way,
            args.k_shot,
            args.q_query,
            args.seed,
            spec.get("class_sampling", "balanced"),
        )
        loader = DataLoader(
            EpisodicDataset(
                train_records,
                args.image_size,
                spec["representation"],
                spec.get("online_augmentation", True),
            ),
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            RecordDataset(train_records, args.image_size, spec["representation"], True, args.seed),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.workers,
            pin_memory=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.10)
    best_score = -1.0
    best_epoch = 0
    epoch_rows = []
    training_seconds = 0.0
    print(
        f"START {args.experiment} train={len(train_records)} val={len(val_records)} test=deferred "
        f"params={parameter_count(model)} device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if spec["training"] == "episodic":
            train_metrics = train_episodic_epoch(model, loader, optimizer, device, spec.get("loss", "proto_only"))
        else:
            train_metrics = train_supervised_epoch(model, loader, optimizer, device)
        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        val_metrics, _ = evaluate(
            model,
            val_records,
            prototype_records,
            spec,
            args,
            maps,
            val_lookup,
            original_train_counts,
            device,
        )
        scheduler.step()
        score = float(val_metrics["macro_f1"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": model.checkpoint_config(),
                    "label_maps": maps,
                    "experiment": args.experiment,
                    "spec": spec,
                    "args": vars(args),
                    "dataset_sha256": DATASET_SHA256,
                    "best_score": best_score,
                },
                output_dir / "best.pt",
            )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_top3": val_metrics["top3"],
            "validation_base_accuracy": val_metrics["base_accuracy"],
            "validation_modifier_accuracy": val_metrics["modifier_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "training_seconds": elapsed,
            "best": False,
        }
        epoch_rows.append(row)
        mark_best_epoch(epoch_rows, best_epoch)
        write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
        print(
            f"EPOCH {epoch}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_top3={val_metrics['top3']:.4f} best={'yes' if improved else 'no'} seconds={elapsed:.1f}",
            flush=True,
        )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    mark_best_epoch(epoch_rows, int(checkpoint["epoch"]))
    write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    checkpoint_hash = sha256(output_dir / "best.pt")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    if args.skip_test:
        selection = {
            "selected_checkpoint_epoch": int(checkpoint["epoch"]),
            "best_validation_macro_f1": float(checkpoint["best_score"]),
            "selection_metric": "validation_macro_f1",
            "test_access": "not_accessed",
            "dataset_sha256": DATASET_SHA256,
            "checkpoint_sha256": checkpoint_hash,
            "seed": args.seed,
            "experiment": args.experiment,
            "model": spec["model"],
            "leakage_audit_status": frozen["leakage_status"],
        }
        (output_dir / "selection_summary.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"SELECTED {args.experiment} epoch={checkpoint['epoch']} "
            f"validation_macro_f1={float(checkpoint['best_score']):.4f} "
            f"checkpoint_sha256={checkpoint_hash} test=not_accessed",
            flush=True,
        )
        return
    val_metrics, val_predictions = evaluate(
        model, val_records, prototype_records, spec, args, maps, val_lookup, original_train_counts, device
    )
    test_records = build_image_records_from_manifest(split_dir / "test_manifest.csv", maps, project_root, "test")
    _, test_lookup = manifest_rows(split_dir / "test_manifest.csv", project_root)
    test_metrics, test_predictions = evaluate(
        model, test_records, prototype_records, spec, args, maps, test_lookup, original_train_counts, device
    )
    write_csv(output_dir / "predictions_validation.csv", val_predictions)
    write_csv(output_dir / "predictions_test.csv", test_predictions)
    result = {
        "status": "PASS",
        "experiment_id": args.experiment,
        "model": spec["model"],
        "input_representation": spec["representation"],
        "description": spec["description"],
        "dataset_sha256": DATASET_SHA256,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "validation": val_metrics,
        "test": test_metrics,
        "training_seconds": training_seconds,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(output_dir / "best.pt"),
        "protocol_deviations": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"TEST {args.experiment} accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} top3={test_metrics['top3']:.4f} "
        f"best_epoch={best_epoch} checkpoint_sha256={checkpoint_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
