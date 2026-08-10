from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.soft_data import (
    SoftFeatureDataset as Data,
    build_image_records,
    build_training_label_maps,
    grouped_split,
    soft_features,
)
from torch.utils.data import DataLoader
from scripts.train_rahas_spatial_proto_v1 import build_prototypes
from src.ocr.rahas_spatial_proto import build_rahas_spatial_proto_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spatial-prototype OCR over a character manifest.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--characters", type=Path, default=Path("datasets/prepared/12_ocr_soft_resized_v1/characters"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-per-class", type=int, default=0, help="Zero keeps every training exemplar.")
    parser.add_argument("--memory-mode", choices=("centroid", "exemplar"), default="exemplar")
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument("--distance-margin", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def select_prototype_records(records, per_class: int, seed: int):
    groups = defaultdict(list)
    for record in records:
        groups[record.full_idx].append(record)
    rng = random.Random(seed + 91)
    selected = []
    for class_records in groups.values():
        count = min(per_class, len(class_records)) if per_class > 0 else len(class_records)
        selected.extend(rng.sample(class_records, count))
    return selected


def encode_exemplar_memory(model, records, image_size: int, device, workers: int):
    loader = DataLoader(Data(records, image_size, False, 0), batch_size=128, shuffle=False, num_workers=workers)
    embeddings = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            embeddings.append(model(batch["image"].to(device))["embedding"])
            labels.append(batch["full"].to(device))
    return torch.cat(embeddings), torch.cat(labels)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_rahas_spatial_proto_from_checkpoint(checkpoint).to(device).eval()
    maps = build_training_label_maps(args.characters)
    records = build_image_records(args.characters, maps)
    seed = int(checkpoint.get("args", {}).get("seed", 2026))
    train_records, _, _ = grouped_split(records, seed)
    support = select_prototype_records(train_records, args.prototype_per_class, seed)
    image_size = int(checkpoint["args"]["image_size"])
    if args.memory_mode == "exemplar":
        prototypes, exemplar_labels = encode_exemplar_memory(model, support, image_size, device, args.workers)
        prototype_indices = torch.unique(exemplar_labels, sorted=True)
    else:
        prototypes, prototype_indices = build_prototypes(model, support, image_size, device, args.workers)
    labels = checkpoint["label_maps"]["idx_to_full_label"]
    threshold = args.distance_threshold
    if threshold is None:
        threshold = float(checkpoint["val_metrics"]["known_distance_p95"]) * 1.25
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    page = Image.open(args.image).convert("RGB")
    tensors = []
    normal_rows = []
    i_label = "\u0907"
    for row_index, row in enumerate(rows):
        if row.get("grouping_reason") == "i_three_dot":
            row.update(
                raw_predicted_label=i_label,
                predicted_label=i_label,
                confidence="1.000000",
                prototype_distance="0.000000",
                distance_margin="1.000000",
                top3=i_label,
                ocr_status="geometry_confirmed",
            )
            continue
        box = tuple(int(row[key]) for key in ("x0", "y0", "x1", "y1"))
        tensors.append(soft_features(page.crop(box).convert("L"), image_size, None, False))
        normal_rows.append(row_index)
    with torch.inference_mode():
        for start in range(0, len(tensors), 128):
            images = torch.stack(tensors[start : start + 128]).to(device)
            embedding = model(images)["embedding"]
            memory_distances = torch.cdist(embedding, prototypes).square()
            if args.memory_mode == "exemplar":
                distances = torch.stack(
                    [memory_distances[:, exemplar_labels == class_index].amin(1) for class_index in prototype_indices],
                    dim=1,
                )
            else:
                distances = memory_distances
            values, local_indices = distances.topk(3, largest=False)
            probabilities = torch.softmax(-distances / 0.15, dim=1)
            for local_index in range(len(images)):
                row = rows[normal_rows[start + local_index]]
                class_indices = prototype_indices[local_indices[local_index]].tolist()
                predicted = [labels[index] for index in class_indices]
                nearest = float(values[local_index, 0])
                margin = float(values[local_index, 1] - values[local_index, 0])
                confidence = float(probabilities[local_index, local_indices[local_index, 0]])
                accepted = nearest <= threshold and margin >= args.distance_margin
                row.update(
                    raw_predicted_label=predicted[0],
                    predicted_label=predicted[0] if accepted else "_",
                    confidence=f"{confidence:.6f}",
                    prototype_distance=f"{nearest:.6f}",
                    distance_margin=f"{margin:.6f}",
                    top3="|".join(predicted),
                    ocr_status="accepted" if accepted else "unknown_distance",
                )
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output / "ocr_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    accepted = sum(row["predicted_label"] != "_" for row in rows)
    raw_text = " ".join(row["raw_predicted_label"] for row in rows)
    accepted_text = " ".join(row["predicted_label"] for row in rows)
    (args.output / "ocr_raw_output.txt").write_text(raw_text, encoding="utf-8")
    (args.output / "ocr_output.txt").write_text(accepted_text, encoding="utf-8")
    overlay = page.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, row in enumerate(rows, start=1):
        box = tuple(int(row[key]) for key in ("x0", "y0", "x1", "y1"))
        ok = row["predicted_label"] != "_"
        draw.rectangle(box, outline=(0, 190, 90) if ok else (235, 150, 0), width=2)
        draw.text((box[0] + 1, box[1] + 1), str(index), fill=(0, 90, 45) if ok else (145, 75, 0), font=font)
    overlay.save(args.output / "ocr_overlay.png")
    print(
        f"device={device} characters={len(rows)} accepted={accepted} unknown={len(rows)-accepted} "
        f"threshold={threshold:.4f} memory_mode={args.memory_mode} references={len(support)} "
        f"checkpoint_epoch={checkpoint.get('epoch')}",
        flush=True,
    )
    print(f"output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
