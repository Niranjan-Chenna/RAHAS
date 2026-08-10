from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from infer_character_alpha_v1 import compose_outputs, input_features, predict_alpha, to_image as alpha_to_image

from segment_estampage_lines_v1 import (
    build_interior_mask,
    dynamic_stroke_map,
    local_relative_depth,
    resize_for_work,
    save_converted_variants,
    suppress_interior_boundaries,
    to_image,
)


def add_project_root(root: Path) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


@dataclass
class Component:
    label: int
    area: int
    x: float
    y: float
    x0: int
    y0: int
    x1: int
    y1: int
    mean_depth: float
    mean_alpha: float = 0.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def density(self) -> float:
        return self.area / max(1, self.width * self.height)

    @property
    def roundness(self) -> float:
        return min(self.width, self.height) / max(self.width, self.height, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment character candidates from an estampage page.")
    parser.add_argument("--input", type=Path, default=Path("datasets/original_estampages/negative_brahmi_estampages_page-0005.jpg"))
    parser.add_argument("--output", type=Path, default=Path("results/segmentation/page_0005_characters_v1"))
    parser.add_argument("--preview", type=Path, default=Path("results/segmentation/previews/page_0005_characters_v1.png"))
    parser.add_argument("--work_max_side", type=int, default=2304)
    parser.add_argument("--crop_pad", type=int, default=12)
    parser.add_argument("--min_main_area", type=int, default=24)
    parser.add_argument("--min_dot_area", type=int, default=8)
    parser.add_argument("--max_component_area", type=int, default=8500)
    parser.add_argument("--max_component_width", type=int, default=210)
    parser.add_argument("--max_component_height", type=int, default=180)
    parser.add_argument("--min_depth", type=float, default=0.065)
    parser.add_argument("--min_dot_depth", type=float, default=0.085)
    parser.add_argument("--neighbor_x", type=int, default=90)
    parser.add_argument("--neighbor_y", type=int, default=58)
    parser.add_argument("--support_threshold", type=float, default=42.0)
    parser.add_argument("--dot_attach_x", type=int, default=58)
    parser.add_argument("--dot_attach_y", type=int, default=42)
    parser.add_argument("--conversion_max_side", type=int, default=700)
    parser.add_argument("--max_chars", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path("pipeline/checkpoints/character_alpha_v1_fast/best.pt"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--alpha_threshold", type=float, default=0.18)
    parser.add_argument("--stroke_alpha_floor", type=float, default=0.055)
    parser.add_argument("--min_alpha_keep", type=float, default=0.10)
    parser.add_argument("--final_alpha_floor", type=float, default=0.08)
    parser.add_argument("--candidate_mode", choices=["recall", "neural_first"], default="recall")
    parser.add_argument("--post_filter_mode", choices=["off", "light", "strict", "matra_floor", "letter_recall"], default="light")
    parser.add_argument("--matra_dot_area", type=int, default=10)
    parser.add_argument("--matra_dot_depth_ratio", type=float, default=0.58)
    parser.add_argument("--adaptive_noise_intensity", action="store_true")
    parser.add_argument("--noise_low_per_mpix", type=float, default=350.0)
    parser.add_argument("--noise_high_per_mpix", type=float, default=1800.0)
    parser.add_argument("--darken_strength", type=float, default=0.52)
    parser.add_argument("--ocr_darken", type=float, default=0.74)
    return parser.parse_args()


def load_alpha_model(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.device]:
    root = args.root.resolve()
    add_project_root(Path(__file__).resolve().parents[1])
    from src.models.character_alpha_net import CharacterAlphaNet

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    base_channels = int(checkpoint_args.get("base_channels", 24))
    activation = str(checkpoint_args.get("activation", "gelu"))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = CharacterAlphaNet(base_channels=base_channels, activation=activation).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = [0]

    def add(self) -> int:
        label = len(self.parent)
        self.parent.append(label)
        return label

    def find(self, label: int) -> int:
        while self.parent[label] != label:
            self.parent[label] = self.parent[self.parent[label]]
            label = self.parent[label]
        return label

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def row_runs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask_row.astype(np.uint8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == 255) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end >= start]


def connected_components(mask: np.ndarray, depth: np.ndarray, alpha: np.ndarray | None = None) -> tuple[list[Component], np.ndarray]:
    height, width = mask.shape
    uf = UnionFind()
    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        current: list[tuple[int, int, int]] = []
        for x0, x1 in row_runs(mask[y]):
            label = uf.add()
            for px0, px1, plabel in previous:
                if px1 < x0 - 1:
                    continue
                if px0 > x1 + 1:
                    break
                uf.union(label, plabel)
            current.append((x0, x1, label))
            runs.append((y, x0, x1, label))
        previous = current

    depth_prefix = np.cumsum(depth, axis=1, dtype=np.float64)
    alpha_prefix = np.cumsum(alpha if alpha is not None else np.zeros_like(depth), axis=1, dtype=np.float64)
    stats: dict[int, list[float]] = {}
    for y, x0, x1, label in runs:
        root = uf.find(label)
        count = x1 - x0 + 1
        depth_sum = float(depth_prefix[y, x1] - (depth_prefix[y, x0 - 1] if x0 else 0.0))
        alpha_sum = float(alpha_prefix[y, x1] - (alpha_prefix[y, x0 - 1] if x0 else 0.0))
        if root not in stats:
            stats[root] = [0.0, 0.0, 0.0, float(x0), float(y), float(x1), float(y), 0.0, 0.0]
        item = stats[root]
        item[0] += count
        item[1] += count * (x0 + x1) / 2.0
        item[2] += count * y
        item[3] = min(item[3], x0)
        item[4] = min(item[4], y)
        item[5] = max(item[5], x1)
        item[6] = max(item[6], y)
        item[7] += depth_sum
        item[8] += alpha_sum

    comps: list[Component] = []
    labels = np.zeros((height, width), dtype=np.int32)
    for y, x0, x1, label in runs:
        labels[y, x0 : x1 + 1] = uf.find(label)
    for label, (area_f, xsum, ysum, x0, y0, x1, y1, depth_sum, alpha_sum) in stats.items():
        area = int(area_f)
        comps.append(
            Component(
                label=label,
                area=area,
                x=xsum / max(area_f, 1.0),
                y=ysum / max(area_f, 1.0),
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                mean_depth=depth_sum / max(area_f, 1.0),
                mean_alpha=alpha_sum / max(area_f, 1.0),
            )
        )
    return comps, labels


def component_score(comp: Component) -> float:
    return comp.area * max(0.03, comp.mean_depth) * max(0.4, comp.density)


def is_border_or_strip(comp: Component, width: int, height: int) -> bool:
    edge = comp.x0 <= 2 or comp.y0 <= 2 or comp.x1 >= width - 3 or comp.y1 >= height - 3
    if edge and (comp.width > width * 0.18 or comp.height > height * 0.18):
        return True
    if comp.width > comp.height * 10 and comp.height <= 8:
        return True
    if comp.height > comp.width * 10 and comp.width <= 8:
        return True
    return False


def has_neighbor_support(comp: Component, comps: list[Component], args: argparse.Namespace) -> bool:
    support = 0.0
    for other in comps:
        if other.label == comp.label:
            continue
        if abs(other.x - comp.x) <= args.neighbor_x and abs(other.y - comp.y) <= args.neighbor_y:
            support += min(component_score(other), 24.0)
    return support >= args.support_threshold


def classify_components(comps: list[Component], width: int, height: int, args: argparse.Namespace) -> tuple[list[Component], list[Component], list[Component]]:
    prelim: list[Component] = []
    rejected: list[Component] = []
    for comp in comps:
        if comp.area > args.max_component_area or comp.width > args.max_component_width or comp.height > args.max_component_height:
            rejected.append(comp)
            continue
        if is_border_or_strip(comp, width, height):
            rejected.append(comp)
            continue
        if comp.density < 0.025:
            rejected.append(comp)
            continue
        neural_rescue = (
            comp.mean_alpha >= args.min_alpha_keep
            and comp.mean_depth >= args.min_depth * 0.65
            and comp.area >= max(args.min_dot_area, 14)
            and max(comp.width, comp.height) >= 6
        )
        if comp.mean_depth < args.min_depth and comp.area < 70 and not neural_rescue:
            rejected.append(comp)
            continue
        prelim.append(comp)

    mains: list[Component] = []
    dots: list[Component] = []
    for comp in prelim:
        if comp.area >= args.min_main_area and comp.height >= 5 and comp.width >= 3:
            continuous = max(comp.width, comp.height) >= 9 or comp.area >= 70
            neural_character = comp.mean_alpha >= args.min_alpha_keep and comp.area >= args.min_main_area and continuous
            if comp.area >= 90 or neural_character or has_neighbor_support(comp, prelim, args):
                mains.append(comp)
            else:
                rejected.append(comp)
            continue
        if (
            comp.area >= args.min_dot_area
            and comp.mean_depth >= args.min_dot_depth
            and comp.roundness >= 0.35
            and comp.width <= 32
            and comp.height <= 32
        ):
            dots.append(comp)
        else:
            rejected.append(comp)
    return mains, dots, rejected


def estimate_noise_intensity(comps: list[Component], shape: tuple[int, int], args: argparse.Namespace) -> tuple[float, float, int]:
    image_mpix = max((shape[0] * shape[1]) / 1_000_000.0, 1e-6)
    likely_speckles = [
        comp
        for comp in comps
        if comp.area < args.matra_dot_area
        and max(comp.width, comp.height) <= 8
        and comp.mean_alpha < args.min_alpha_keep * 0.7
        and comp.mean_depth >= args.min_depth * 0.35
    ]
    speckles_per_mpix = len(likely_speckles) / image_mpix
    intensity = float(
        np.clip(
            (speckles_per_mpix - args.noise_low_per_mpix)
            / max(args.noise_high_per_mpix - args.noise_low_per_mpix, 1e-6),
            0.0,
            1.0,
        )
    )
    return intensity, speckles_per_mpix, len(likely_speckles)


def attach_dots(mains: list[Component], dots: list[Component], args: argparse.Namespace) -> dict[int, list[Component]]:
    attached: dict[int, list[Component]] = {comp.label: [] for comp in mains}
    for dot in dots:
        best: Component | None = None
        best_cost = 1e9
        for main in mains:
            dx = max(0.0, abs(dot.x - main.x) - max(main.width * 0.5, args.dot_attach_x))
            dy = max(0.0, abs(dot.y - main.y) - max(main.height * 0.5, args.dot_attach_y))
            cost = dx + dy * 1.4
            if cost < best_cost:
                best = main
                best_cost = cost
        if best is not None and best_cost <= args.dot_attach_x:
            attached[best.label].append(dot)
    return attached


def component_bbox(comp: Component, attached: list[Component], width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = comp.x0, comp.y0, comp.x1, comp.y1
    for dot in attached:
        x0 = min(x0, dot.x0)
        y0 = min(y0, dot.y0)
        x1 = max(x1, dot.x1)
        y1 = max(y1, dot.y1)
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad + 1), min(height, y1 + pad + 1)


def make_final_keep_mask(
    combined: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    label_map: np.ndarray,
    mains: list[Component],
    dots_by_main: dict[int, list[Component]],
    args: argparse.Namespace,
) -> np.ndarray:
    keep_labels: set[int] = set()
    for comp in mains:
        keep_labels.add(comp.label)
        for dot in dots_by_main.get(comp.label, []):
            keep_labels.add(dot.label)
    keep = np.isin(label_map, list(keep_labels)) if keep_labels else np.zeros_like(combined, dtype=bool)
    pixel_gate = combined & ((alpha >= args.final_alpha_floor) | (depth >= args.min_depth * 0.62))
    return keep & pixel_gate


def make_letter_recall_keep_mask(
    comps: list[Component],
    label_map: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> np.ndarray:
    accepted: set[int] = set()
    for comp in comps:
        if is_border_or_strip(comp, width, height):
            continue

        too_tiny = comp.area < args.matra_dot_area and max(comp.width, comp.height) <= 6
        weak_tiny = too_tiny and comp.mean_depth < args.min_depth * 1.15 and comp.mean_alpha < args.min_alpha_keep
        if weak_tiny:
            continue

        matra_floor = comp.area >= args.matra_dot_area and comp.mean_depth >= args.min_depth * args.matra_dot_depth_ratio
        broken_letter_piece = (
            comp.area >= max(4, args.matra_dot_area - 5)
            and max(comp.width, comp.height) >= 7
            and comp.mean_depth >= args.min_depth * 0.42
        )
        neural_piece = comp.area >= max(4, args.matra_dot_area - 6) and comp.mean_alpha >= args.min_alpha_keep * 0.35
        dark_dot = (
            comp.area >= max(args.min_dot_area, args.matra_dot_area - 2)
            and comp.mean_depth >= args.min_dot_depth * 0.55
            and comp.width <= 46
            and comp.height <= 46
        )
        if matra_floor or broken_letter_piece or neural_piece or dark_dot:
            accepted.add(comp.label)
    return np.isin(label_map, list(accepted)) if accepted else np.zeros_like(label_map, dtype=bool)


def post_filter_keep_mask(
    keep: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    mains: list[Component],
    args: argparse.Namespace,
) -> np.ndarray:
    comps, label_map = connected_components(keep, depth, alpha)
    main_centers = [(comp.x, comp.y, comp.width, comp.height) for comp in mains]
    noise_intensity = float(getattr(args, "_noise_intensity", 0.0)) if args.adaptive_noise_intensity else 0.0
    dot_area_floor = int(round(args.matra_dot_area + noise_intensity * 6.0))
    dot_depth_ratio = args.matra_dot_depth_ratio + noise_intensity * 0.26
    isolated_density_floor = 0.025 + noise_intensity * 0.035
    accepted: set[int] = set()
    for comp in comps:
        if args.post_filter_mode == "matra_floor":
            near_main = False
            for mx, my, mw, mh in main_centers:
                if abs(comp.x - mx) <= max(args.dot_attach_x, mw * 0.85) and abs(comp.y - my) <= max(args.dot_attach_y, mh * 0.95):
                    near_main = True
                    break
            continuous_letter = (
                max(comp.width, comp.height) >= 7
                and comp.area >= max(6, args.matra_dot_area - 3)
                and comp.mean_depth >= args.min_depth * max(args.matra_dot_depth_ratio, 0.50)
                and comp.density >= 0.018
            )
            matra_sized_mark = (
                (near_main or noise_intensity < 0.28 or max(comp.width, comp.height) >= 13)
                and comp.area >= dot_area_floor
                and comp.mean_depth >= args.min_depth * dot_depth_ratio
                and comp.density >= isolated_density_floor
            )
            valid_near_dot = (
                near_main
                and comp.area >= max(args.min_dot_area, dot_area_floor)
                and comp.mean_depth >= args.min_dot_depth * (0.62 + noise_intensity * 0.16)
                and comp.width <= 42
                and comp.height <= 42
            )
            if continuous_letter or matra_sized_mark or valid_near_dot:
                accepted.add(comp.label)
            continue

        if args.post_filter_mode == "light":
            near_main = False
            for mx, my, mw, mh in main_centers:
                if abs(comp.x - mx) <= max(args.dot_attach_x, mw * 0.75) and abs(comp.y - my) <= max(args.dot_attach_y, mh * 0.85):
                    near_main = True
                    break
            keep_continuous = (
                comp.area >= max(args.min_main_area, 18)
                and (max(comp.width, comp.height) >= 10 or comp.area >= 42)
                and comp.mean_depth >= args.min_depth * 0.72
                and comp.density >= 0.04
                and (comp.mean_alpha >= args.min_alpha_keep * 0.35 or comp.area >= 46)
            )
            keep_dot = (
                near_main
                and comp.area >= max(args.min_dot_area, 9)
                and comp.mean_depth >= args.min_dot_depth * 0.78
                and comp.roundness >= 0.28
                and comp.width <= 36
                and comp.height <= 36
                and (comp.mean_alpha >= args.min_alpha_keep * 0.25 or comp.mean_depth >= args.min_dot_depth)
            )
            if keep_continuous or keep_dot:
                accepted.add(comp.label)
            continue

        continuous = max(comp.width, comp.height) >= 8 or comp.area >= 70
        strong_main = (
            comp.area >= args.min_main_area
            and comp.mean_depth >= args.min_depth * 0.72
            and comp.density >= 0.035
            and continuous
        )
        neural_main = comp.area >= args.min_main_area and comp.mean_alpha >= args.min_alpha_keep * 1.2 and continuous
        near_main = False
        for mx, my, mw, mh in main_centers:
            if abs(comp.x - mx) <= max(args.dot_attach_x, mw * 0.55) and abs(comp.y - my) <= max(args.dot_attach_y, mh * 0.75):
                near_main = True
                break
        valid_dot = (
            near_main
            and comp.area >= max(args.min_dot_area, 12)
            and comp.mean_depth >= args.min_dot_depth * 0.82
            and comp.roundness >= 0.32
            and comp.width <= 34
            and comp.height <= 34
        )
        if strong_main or neural_main or valid_dot:
            accepted.add(comp.label)
    return np.isin(label_map, list(accepted)) if accepted else np.zeros_like(keep, dtype=bool)


def save_character_outputs(
    image: Image.Image,
    depth: np.ndarray,
    stroke: np.ndarray,
    alpha: np.ndarray,
    combined: np.ndarray,
    final_keep: np.ndarray,
    cleaned_soft: np.ndarray,
    cleaned_dark: np.ndarray,
    mains: list[Component],
    dots_by_main: dict[int, list[Component]],
    rejected: list[Component],
    args: argparse.Namespace,
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    rows: list[dict[str, int | float]] = []

    all_ordered = sorted(mains, key=lambda comp: (comp.y, comp.x))
    for comp in all_ordered:
        attached = dots_by_main.get(comp.label, [])
        x0, y0, x1, y1 = component_bbox(comp, attached, image.width, image.height, args.crop_pad)
        draw.rectangle((x0, y0, x1, y1), outline=(0, 220, 80), width=1)

    ordered = all_ordered[: args.max_chars] if args.max_chars else all_ordered
    for index, comp in enumerate(ordered, start=1):
        attached = dots_by_main.get(comp.label, [])
        x0, y0, x1, y1 = component_bbox(comp, attached, image.width, image.height, args.crop_pad)
        char_dir = args.output / f"char_{index:04d}"
        char_dir.mkdir(parents=True, exist_ok=True)
        crop = image.crop((x0, y0, x1, y1))
        crop.save(char_dir / "input.png")
        to_image(depth[y0:y1, x0:x1]).save(char_dir / "depth.png")
        to_image(stroke[y0:y1, x0:x1].astype(np.float32)).save(char_dir / "mask.png")
        alpha_to_image(alpha[y0:y1, x0:x1]).save(char_dir / "neural_alpha.png")
        to_image(final_keep[y0:y1, x0:x1].astype(np.float32)).save(char_dir / "final_keep_mask.png")
        to_image(cleaned_soft[y0:y1, x0:x1]).save(char_dir / "restored_soft.png")
        to_image(cleaned_dark[y0:y1, x0:x1]).save(char_dir / "restored_dark_ocr.png")
        save_converted_variants(crop, char_dir / "converted", args.conversion_max_side)
        if index <= 999:
            draw.text((x0, max(0, y0 - 9)), str(index), fill=(0, 220, 80), font=font)
        rows.append(
            {
                "char": index,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "area": comp.area,
                "mean_depth": round(comp.mean_depth, 4),
                "mean_alpha": round(comp.mean_alpha, 4),
                "attached_dots": len(attached),
            }
        )

    for comp in rejected[:3000]:
        if comp.area >= 4:
            draw.point((int(comp.x), int(comp.y)), fill=(255, 170, 0))

    overlay.save(args.output / "character_overlay.png")
    to_image(stroke.astype(np.float32)).save(args.output / "stroke_candidate.png")
    alpha_to_image(alpha).save(args.output / "neural_alpha.png")
    to_image(combined.astype(np.float32)).save(args.output / "combined_candidate.png")
    to_image(final_keep.astype(np.float32)).save(args.output / "final_keep_mask.png")
    to_image(cleaned_soft).save(args.output / "restored_soft.png")
    to_image(cleaned_dark).save(args.output / "restored_dark_ocr.png")
    with (args.output / "noise_profile.txt").open("w", encoding="utf-8") as file:
        file.write(f"adaptive_noise_intensity={bool(args.adaptive_noise_intensity)}\n")
        file.write(f"noise_intensity={float(getattr(args, '_noise_intensity', 0.0)):.6f}\n")
        file.write(f"speckles_per_mpix={float(getattr(args, '_speckles_per_mpix', 0.0)):.3f}\n")
        file.write(f"speckle_count={int(getattr(args, '_speckle_count', 0))}\n")
    if rows:
        with (args.output / "character_manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    make_contact_sheet(args.output, len(rows), args.preview)


def make_contact_sheet(output: Path, count: int, preview: Path) -> None:
    overlay = Image.open(output / "character_overlay.png").convert("RGB")
    overlay.thumbnail((1200, 850), Image.Resampling.LANCZOS)
    channels: list[tuple[str, Image.Image]] = []
    for name in ["neural_alpha", "stroke_candidate", "combined_candidate", "final_keep_mask", "restored_soft", "restored_dark_ocr"]:
        path = output / f"{name}.png"
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((280, 200), Image.Resampling.LANCZOS)
            channels.append((name, image))
    thumbs = []
    for index in range(1, min(count, 160) + 1):
        path = output / f"char_{index:04d}" / "converted_white_on_black.png"
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((92, 92), Image.Resampling.LANCZOS)
            thumbs.append((index, image))
    tile = 104
    label_h = 14
    cols = 12
    rows = int(math.ceil(len(thumbs) / cols)) if thumbs else 0
    margin = 12
    channel_w = margin * 2 + max(0, len(channels)) * 292
    sheet_w = max(1200 + margin * 2, margin * 2 + cols * tile, channel_w)
    channel_h = 230 if channels else 0
    sheet_h = margin * 4 + channel_h + overlay.height + rows * (tile + label_h)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (248, 248, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y_cursor = margin
    if channels:
        x = margin
        for name, image in channels:
            sheet.paste(image, (x, y_cursor))
            draw.rectangle((x, y_cursor, x + 280, y_cursor + 200), outline=(210, 210, 205))
            draw.text((x, y_cursor + 204), name, fill=(30, 30, 30), font=font)
            x += 292
        y_cursor += channel_h
    sheet.paste(overlay, ((sheet_w - overlay.width) // 2, y_cursor))
    y0 = y_cursor + margin + overlay.height
    for i, (index, image) in enumerate(thumbs):
        col = i % cols
        row = i // cols
        x = margin + col * tile
        y = y0 + row * (tile + label_h)
        sheet.paste(image, (x + (tile - image.width) // 2, y + (tile - image.height) // 2))
        draw.rectangle((x, y, x + tile, y + tile), outline=(210, 210, 205))
        draw.text((x + 3, y + tile + 1), f"{index:04d}", fill=(30, 30, 30), font=font)
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview)


def main() -> None:
    args = parse_args()
    source = Image.open(args.input).convert("L")
    work, scale = resize_for_work(source, args.work_max_side)
    model, device = load_alpha_model(args)
    gray_arr, model_darkness, features = input_features(work)
    alpha = predict_alpha(model, features, device, args.tile, args.overlap)
    _, bg, depth = local_relative_depth(work)
    depth = np.maximum(depth, model_darkness * 0.9)
    interior = build_interior_mask(bg, argparse.Namespace(interior_threshold=0.30, interior_erode=25))
    stroke, score = dynamic_stroke_map(
        depth,
        interior,
        argparse.Namespace(
            window_step=48,
            window_width=96,
            min_depth=0.055,
            faint_min_depth=0.045,
            depth_percentile=86.0,
            faint_depth_percentile=80.0,
            strong_density_threshold=0.075,
            faint_density_threshold=0.055,
        ),
    )
    score = suppress_interior_boundaries(score, interior, argparse.Namespace(interior_boundary_margin=36))
    stroke = stroke & (score > 0)
    neural_pixels = (alpha >= args.alpha_threshold) & (depth >= args.min_depth * 0.42)
    if args.candidate_mode == "recall":
        stroke_pixels = stroke & (depth >= args.min_depth * 0.54)
    else:
        stroke_pixels = stroke & (alpha >= args.stroke_alpha_floor) & (depth >= args.min_depth * 0.82)
    combined = (neural_pixels | stroke_pixels) & interior
    comps, label_map = connected_components(combined, depth, alpha)
    noise_intensity, speckles_per_mpix, speckle_count = estimate_noise_intensity(comps, combined.shape, args)
    args._noise_intensity = noise_intensity
    args._speckles_per_mpix = speckles_per_mpix
    args._speckle_count = speckle_count
    mains, dots, rejected = classify_components(comps, work.width, work.height, args)
    dots_by_main = attach_dots(mains, dots, args)
    if args.post_filter_mode == "letter_recall":
        final_keep = make_letter_recall_keep_mask(comps, label_map, work.width, work.height, args)
    else:
        final_keep = make_final_keep_mask(combined, depth, alpha, label_map, mains, dots_by_main, args)
        if args.post_filter_mode != "off":
            final_keep = post_filter_keep_mask(final_keep, depth, alpha, mains, args)
    final_alpha = np.where(final_keep, np.maximum(alpha, np.clip(depth * 1.25, 0.0, 1.0)), 0.0).astype(np.float32)
    cleaned_soft, cleaned_dark = compose_outputs(gray_arr, model_darkness, final_alpha, args)
    save_character_outputs(
        work,
        depth,
        stroke,
        alpha,
        combined,
        final_keep,
        cleaned_soft,
        cleaned_dark,
        mains,
        dots_by_main,
        rejected + dots,
        args,
    )
    print(f"input: {args.input}")
    print(f"work size: {work.width}x{work.height} scale={scale:.4f}")
    print(f"device: {device}")
    print(f"components: {len(comps)}")
    print(f"characters: {len(mains)}")
    print(f"dots: {len(dots)}")
    print(f"noise speckles/mpix: {speckles_per_mpix:.1f} count={speckle_count} intensity={noise_intensity:.3f}")
    print(f"output: {args.output}")
    print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
