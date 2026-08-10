from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass
class Component:
    label: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int
    mean_depth: float
    mean_alpha: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def density(self) -> float:
        return self.area / max(1, self.width * self.height)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse estampage stroke-continuity restoration.")
    parser.add_argument("--input", type=Path, default=Path("datasets/original_estampages/negative_brahmi_estampages_page-0013.jpg"))
    parser.add_argument("--output", type=Path, default=Path("results/restoration/page_0013_sparse_stroke_v1"))
    parser.add_argument("--preview", type=Path, default=Path("results/restoration/previews/page_0013_sparse_stroke_v1.png"))
    parser.add_argument("--work_max_side", type=int, default=0)
    parser.add_argument("--background_radius", type=int, default=91)
    parser.add_argument("--candidate_percentile", type=float, default=76.0)
    parser.add_argument("--min_depth", type=float, default=0.035)
    parser.add_argument("--weak_depth", type=float, default=0.022)
    parser.add_argument("--matra_dot_area", type=int, default=24)
    parser.add_argument("--tiny_area", type=int, default=8)
    parser.add_argument("--connect_iters", type=int, default=1)
    parser.add_argument("--soft_gain", type=float, default=2.4)
    parser.add_argument("--dark_gain", type=float, default=3.0)
    parser.add_argument("--max_debug_components", type=int, default=2500)
    return parser.parse_args()


def resize_for_work(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0 or max(image.size) <= max_side:
        return image, 1.0
    scale = max_side / float(max(image.size))
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    return resized, scale


def to_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")


def estimate_background(gray: Image.Image, radius: int) -> Image.Image:
    radius = max(9, radius | 1)
    return gray.filter(ImageFilter.MaxFilter(radius)).filter(ImageFilter.GaussianBlur(max(1.0, radius / 5.5)))


def shifted_or(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        acc = out.copy()
        for dy in range(3):
            for dx in range(3):
                acc |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = acc
    return out


def shifted_and(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        acc = np.ones_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                acc &= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = acc
    return out


def close_small_gaps(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    return shifted_and(shifted_or(mask, iterations), iterations)


def row_runs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask_row.astype(np.uint8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == 255) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end >= start]


def connected_components(mask: np.ndarray, depth: np.ndarray, alpha: np.ndarray) -> tuple[list[Component], np.ndarray]:
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
    alpha_prefix = np.cumsum(alpha, axis=1, dtype=np.float64)
    stats: dict[int, list[float]] = {}
    labels = np.zeros((height, width), dtype=np.int32)
    for y, x0, x1, label in runs:
        root = uf.find(label)
        labels[y, x0 : x1 + 1] = root
        count = x1 - x0 + 1
        depth_sum = float(depth_prefix[y, x1] - (depth_prefix[y, x0 - 1] if x0 else 0.0))
        alpha_sum = float(alpha_prefix[y, x1] - (alpha_prefix[y, x0 - 1] if x0 else 0.0))
        if root not in stats:
            stats[root] = [0.0, float(x0), float(y), float(x1), float(y), 0.0, 0.0]
        item = stats[root]
        item[0] += count
        item[1] = min(item[1], x0)
        item[2] = min(item[2], y)
        item[3] = max(item[3], x1)
        item[4] = max(item[4], y)
        item[5] += depth_sum
        item[6] += alpha_sum

    comps: list[Component] = []
    for label, (area_f, x0, y0, x1, y1, depth_sum, alpha_sum) in stats.items():
        area = int(area_f)
        comps.append(
            Component(
                label=label,
                area=area,
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                mean_depth=depth_sum / max(area_f, 1.0),
                mean_alpha=alpha_sum / max(area_f, 1.0),
            )
        )
    return comps, labels


def keep_stroke_components(comps: list[Component], labels: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[Component]]:
    keep_labels: set[int] = set()
    rejected: list[Component] = []
    height, width = labels.shape
    for comp in comps:
        edge = comp.x0 <= 1 or comp.y0 <= 1 or comp.x1 >= width - 2 or comp.y1 >= height - 2
        border_strip = edge and (comp.width > width * 0.18 or comp.height > height * 0.18)
        if border_strip:
            rejected.append(comp)
            continue

        tiny_weak = (
            comp.area < args.matra_dot_area
            and max(comp.width, comp.height) <= 7
            and comp.mean_depth < args.min_depth * 1.4
            and comp.mean_alpha < 0.45
        )
        dust = comp.area <= args.tiny_area and comp.mean_depth < args.min_depth * 2.0
        if tiny_weak or dust:
            rejected.append(comp)
            continue

        stroke_like = (
            comp.area >= args.matra_dot_area
            or max(comp.width, comp.height) >= 9
            or comp.mean_alpha >= 0.35
            or comp.mean_depth >= args.min_depth * 1.8
        )
        if stroke_like:
            keep_labels.add(comp.label)
        else:
            rejected.append(comp)

    keep = np.isin(labels, list(keep_labels)) if keep_labels else np.zeros_like(labels, dtype=bool)
    return keep, rejected


def save_preview(args: argparse.Namespace, channels: list[tuple[str, Image.Image]]) -> None:
    tile_w, tile_h = 260, 180
    label_h = 18
    margin = 12
    sheet = Image.new("RGB", (margin * 2 + len(channels) * tile_w, margin * 2 + tile_h + label_h), (248, 248, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (name, image) in enumerate(channels):
        thumb = image.convert("RGB")
        thumb.thumbnail((tile_w - 10, tile_h), Image.Resampling.LANCZOS)
        x = margin + i * tile_w
        y = margin
        sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y))
        draw.rectangle((x, y, x + tile_w - 4, y + tile_h), outline=(210, 210, 205))
        draw.text((x, y + tile_h + 2), name, fill=(30, 30, 30), font=font)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.preview)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.input).convert("L")
    work, scale = resize_for_work(source, args.work_max_side)
    gray = np.asarray(work, dtype=np.float32) / 255.0
    background = estimate_background(work, args.background_radius)
    bg = np.asarray(background, dtype=np.float32) / 255.0
    depth = np.clip((bg - gray) / np.maximum(bg, 0.08), 0.0, 1.0)
    depth = np.power(depth, 0.82)

    positive = depth[depth > args.weak_depth]
    threshold = max(args.min_depth, float(np.percentile(positive, args.candidate_percentile)) if positive.size else args.min_depth)
    strong = depth >= threshold
    weak = depth >= args.weak_depth
    bridge = close_small_gaps(strong, args.connect_iters) & weak
    candidate = strong | bridge

    alpha = np.clip(depth / max(threshold, 1e-4), 0.0, 1.0)
    comps, labels = connected_components(candidate, depth, alpha)
    keep, rejected = keep_stroke_components(comps, labels, args)
    alpha = np.where(keep, np.clip(depth * args.soft_gain, 0.0, 1.0), 0.0)
    soft_source = np.clip(gray - depth * 0.35, 0.0, 1.0)
    restored_soft = np.clip(1.0 - alpha + soft_source * alpha, 0.0, 1.0)
    dark_source = np.clip(gray - depth * args.dark_gain, 0.0, 1.0)
    restored_dark = np.clip(1.0 - alpha + dark_source * alpha, 0.0, 1.0)

    overlay = work.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for comp in comps[: args.max_debug_components]:
        color = (0, 220, 80) if keep[comp.y0 : comp.y1 + 1, comp.x0 : comp.x1 + 1].any() else (255, 170, 0)
        if comp.area >= args.tiny_area:
            draw.rectangle((comp.x0, comp.y0, comp.x1, comp.y1), outline=color, width=1)

    work.save(args.output / "input.png")
    background.save(args.output / "background.png")
    to_image(depth).save(args.output / "darkness.png")
    to_image(candidate.astype(np.float32)).save(args.output / "stroke_candidate.png")
    to_image(keep.astype(np.float32)).save(args.output / "keep_mask.png")
    to_image(alpha).save(args.output / "soft_alpha.png")
    to_image(restored_soft).save(args.output / "restored_soft.png")
    to_image(restored_dark).save(args.output / "restored_dark_ocr.png")
    overlay.save(args.output / "component_overlay.png")

    with (args.output / "stroke_profile.txt").open("w", encoding="utf-8") as file:
        file.write(f"input={args.input}\n")
        file.write(f"work_size={work.width}x{work.height}\n")
        file.write(f"scale={scale:.6f}\n")
        file.write(f"threshold={threshold:.6f}\n")
        file.write(f"components={len(comps)}\n")
        file.write(f"rejected_components={len(rejected)}\n")
        file.write(f"kept_pixels={int(keep.sum())}\n")

    save_preview(
        args,
        [
            ("input", work),
            ("darkness", to_image(depth)),
            ("candidate", to_image(candidate.astype(np.float32))),
            ("keep_mask", to_image(keep.astype(np.float32))),
            ("restored_soft", to_image(restored_soft)),
            ("restored_dark_ocr", to_image(restored_dark)),
        ],
    )
    print(f"input: {args.input}")
    print(f"work size: {work.width}x{work.height} scale={scale:.4f}")
    print(f"threshold: {threshold:.4f}")
    print(f"components: {len(comps)} rejected={len(rejected)}")
    print(f"output: {args.output}")
    print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
