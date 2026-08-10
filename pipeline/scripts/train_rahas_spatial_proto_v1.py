from __future__ import annotations

import argparse
import csv
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

from src.ocr.soft_data import (
    SoftFeatureDataset as Data,
    build_image_records,
    build_image_records_from_manifest,
    build_training_label_maps,
    grouped_split,
    macro_f1,
    soft_features,
)
from src.ocr.rahas_spatial_proto import RahasSpatialProto, squared_euclidean_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-disjoint spatial prototypical OCR.")
    parser.add_argument("--characters", type=Path, default=Path("datasets/prepared/12_ocr_soft_resized_v1/characters"))
    parser.add_argument("--split-dir", type=Path, help="Directory containing frozen train/validation/test manifests.")
    parser.add_argument("--output", type=Path, default=Path("pipeline/checkpoints/rahas_spatial_proto_v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes-per-epoch", type=int, default=40)
    parser.add_argument("--n-way", type=int, default=24)
    parser.add_argument("--k-shot", type=int, default=2)
    parser.add_argument("--q-query", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prototype-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


class EpisodicDataset(Dataset):
    def __init__(self, records, image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key):
        index, augmentation_seed, local_class, is_support = key
        record = self.records[index]
        image = Image.open(record.path).convert("L")
        features = soft_features(image, self.image_size, random.Random(augmentation_seed), train=True)
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
    def __init__(self, records, episodes: int, n_way: int, k_shot: int, q_query: int, seed: int) -> None:
        self.by_class = defaultdict(list)
        for index, record in enumerate(records):
            self.by_class[record.full_idx].append(index)
        self.classes = sorted(self.by_class)
        self.episodes = episodes
        self.n_way = min(n_way, len(self.classes))
        self.k_shot = k_shot
        self.q_query = q_query
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        self.epoch += 1
        for episode in range(self.episodes):
            selected = rng.sample(self.classes, self.n_way)
            support = []
            query = []
            for local_class, full_class in enumerate(selected):
                candidates = self.by_class[full_class]
                needed = self.k_shot + self.q_query
                chosen = rng.sample(candidates, needed) if len(candidates) >= needed else rng.choices(candidates, k=needed)
                for offset, index in enumerate(chosen[: self.k_shot]):
                    support.append((index, rng.randrange(2**31), local_class, True))
                for offset, index in enumerate(chosen[self.k_shot :]):
                    query.append((index, rng.randrange(2**31), local_class, False))
            yield support + query

    def __len__(self) -> int:
        return self.episodes


def build_prototypes(model, records, image_size: int, device, workers: int):
    loader = DataLoader(Data(records, image_size, False, 0), batch_size=128, shuffle=False, num_workers=workers)
    sums = torch.zeros(model.config.num_full_labels, model.config.embedding_dim, device=device)
    counts = torch.zeros(model.config.num_full_labels, device=device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch["full"].to(device)
            embeddings = model(batch["image"].to(device))["embedding"]
            sums.index_add_(0, labels, embeddings)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    valid = counts > 0
    prototypes = F.normalize(sums[valid] / counts[valid, None], dim=1)
    return prototypes, torch.arange(len(counts), device=device)[valid]


def evaluate(model, train_records, eval_records, image_size: int, device, workers: int):
    prototypes, prototype_labels = build_prototypes(model, train_records, image_size, device, workers)
    loader = DataLoader(Data(eval_records, image_size, False, 1), batch_size=128, shuffle=False, num_workers=workers)
    predictions = []
    targets = []
    distances = []
    base_correct = modifier_correct = seen = top3_correct = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device))
            labels = batch["full"].to(device)
            distance = torch.cdist(output["embedding"], prototypes).square()
            nearest = distance.argmin(1)
            prediction = prototype_labels[nearest]
            top3 = prototype_labels[distance.topk(3, largest=False).indices]
            predictions.append(prediction.cpu())
            targets.append(labels.cpu())
            distances.append(distance.gather(1, nearest[:, None]).squeeze(1).cpu())
            top3_correct += int((top3 == labels[:, None]).any(1).sum())
            base_correct += int((output["base_glyph"].argmax(1) == batch["base"].to(device)).sum())
            modifier_correct += int((output["modifier"].argmax(1) == batch["modifier"].to(device)).sum())
            seen += len(labels)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    known_distance = torch.cat(distances)
    return {
        "accuracy": float((prediction == target).float().mean()),
        "macro_f1": macro_f1(prediction, target, model.config.num_full_labels),
        "top3": top3_correct / seen,
        "base_accuracy": base_correct / seen,
        "modifier_accuracy": modifier_correct / seen,
        "known_distance_p95": float(torch.quantile(known_distance, 0.95)),
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    totals = defaultdict(float)
    seen_queries = 0
    for batch in loader:
        images = batch["image"].to(device)
        local = batch["local"].to(device)
        support_mask = batch["support"].to(device).bool()
        output = model(images)
        embeddings = output["embedding"]
        local_classes = int(local.max()) + 1
        prototypes = torch.stack([embeddings[support_mask & (local == index)].mean(0) for index in range(local_classes)])
        prototypes = F.normalize(prototypes, dim=1)
        query_logits = squared_euclidean_logits(embeddings[~support_mask], prototypes)
        query_targets = local[~support_mask]
        proto_loss = F.cross_entropy(query_logits, query_targets)
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    maps = build_training_label_maps(args.characters)
    if args.split_dir:
        project_root = Path.cwd()
        train_records = build_image_records_from_manifest(
            args.split_dir / "train_manifest.csv", maps, project_root, "train"
        )
        val_records = build_image_records_from_manifest(
            args.split_dir / "validation_manifest.csv", maps, project_root, "validation"
        )
        test_records = build_image_records_from_manifest(
            args.split_dir / "test_manifest.csv", maps, project_root, "test"
        )
        records = train_records + val_records + test_records
        split_description = f"frozen_manifests={args.split_dir}"
    else:
        records = build_image_records(args.characters, maps)
        train_records, val_records, test_records = grouped_split(records, args.seed)
        split_description = "legacy_runtime_grouped_split"
    prototype_groups = defaultdict(list)
    for record in train_records:
        prototype_groups[record.full_idx].append(record)
    prototype_records = []
    prototype_rng = random.Random(args.seed + 91)
    for class_records in prototype_groups.values():
        count = min(args.prototype_per_class, len(class_records))
        prototype_records.extend(prototype_rng.sample(class_records, count))
    sampler = EpisodeSampler(train_records, args.episodes_per_epoch, args.n_way, args.k_shot, args.q_query, args.seed)
    loader = DataLoader(EpisodicDataset(train_records, args.image_size), batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RahasSpatialProto(
        num_full_labels=len(maps["idx_to_full_label"]),
        num_base_glyphs=len(maps["idx_to_base_glyph"]),
        num_modifiers=len(maps["idx_to_modifier"]),
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        grid_size=args.grid_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.10)
    start_epoch = 1
    best_score = (-1.0, -1.0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = tuple(checkpoint.get("best_score", best_score))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    print(
        f"{split_description} total={len(records)} train={len(train_records)} val={len(val_records)} "
        f"test={len(test_records)} classes={len(set(record.full_idx for record in train_records))} device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_epoch(model, loader, optimizer, device)
        train_seconds = time.perf_counter() - epoch_started
        print(f"epoch={epoch} training_complete seconds={train_seconds:.1f}", flush=True)
        val_metrics = evaluate(model, prototype_records, val_records, args.image_size, device, args.workers)
        scheduler.step()
        score = (val_metrics["macro_f1"], val_metrics["accuracy"])
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
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_score": best_score,
        }
        torch.save(state, args.output / "latest.pt")
        if improved:
            torch.save(state, args.output / "best.pt")
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"], "best": improved}
        rows.append(row)
        with (args.output / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"EPOCH {epoch}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"episode_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} top3={val_metrics['top3']:.4f} "
            f"base={val_metrics['base_accuracy']:.4f} modifier={val_metrics['modifier_accuracy']:.4f} "
            f"best={'yes' if improved else 'no'}",
            flush=True,
        )
    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(model, prototype_records, test_records, args.image_size, device, args.workers)
    with (args.output / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_checkpoint_epoch": int(best["epoch"]),
                "best_validation_score": list(best["best_score"]),
                "test_metrics": test_metrics,
                "split_dir": str(args.split_dir) if args.split_dir else None,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"TEST accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f} top3={test_metrics['top3']:.4f}", flush=True)


if __name__ == "__main__":
    main()
