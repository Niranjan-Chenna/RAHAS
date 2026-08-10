from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEBUG_CHANNELS = [
    "input",
    "darkness",
    "candidate_raw",
    "rejected_speckles",
    "rejected_lines",
    "rejected_strips",
    "keep_mask",
    "alpha",
    "cleaned_soft",
    "cleaned_dark_ocr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer neural character-only soft alpha cleanup.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=Path("data/02_real_words/hard_cases/final_test"))
    parser.add_argument("--output", type=Path, default=Path("results/restoration/samples/character_alpha_v1_hard_words"))
    parser.add_argument("--preview", type=Path, default=Path("results/restoration/previews/character_alpha_v1/hard_words_preview.png"))
    parser.add_argument("--checkpoint", type=Path, default=Path("pipeline/checkpoints/character_alpha_v1_fast/best.pt"))
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--work_max_side", type=int, default=1536)
    parser.add_argument("--alpha_threshold", type=float, default=0.28)
    parser.add_argument("--min_keep_area_ratio", type=float, default=0.00012)
    parser.add_argument("--max_speckle_area_ratio", type=float, default=0.000045)
    parser.add_argument("--line_aspect", type=float, default=9.0)
    parser.add_argument("--strip_edge_ratio", type=float, default=0.055)
    parser.add_argument("--darken_strength", type=float, default=0.52)
    parser.add_argument("--ocr_darken", type=float, default=0.74)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--full_canvas", action="store_true")
    return parser.parse_args()


def add_project_root(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.relative_to(folder).as_posix().casefold(),
    )


def to_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")


def local_background(gray: Image.Image, max_pixels: int = 900_000) -> Image.Image:
    original_size = gray.size
    width, height = gray.size
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        small = gray.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
        return local_background(small, max_pixels).resize(original_size, Image.Resampling.BILINEAR)
    radius = max(11, min(71, (min(width, height) // 18) | 1))
    return gray.filter(ImageFilter.MaxFilter(radius)).filter(ImageFilter.GaussianBlur(max(1.0, radius / 4.5)))


def input_features(gray: Image.Image) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    background = np.asarray(local_background(gray), dtype=np.float32) / 255.0
    darkness = np.clip((background - arr) / np.maximum(background, 0.08), 0.0, 1.0)
    darkness = np.power(darkness, 0.85)
    gray_t = torch.from_numpy(arr[None, None, :, :])
    local_mean = F.avg_pool2d(gray_t, 9, stride=1, padding=4)
    local_sq = F.avg_pool2d(gray_t.square(), 9, stride=1, padding=4)
    contrast = ((local_sq - local_mean.square()).clamp_min(0.0).sqrt() * 4.0).clamp(0.0, 1.0)
    features = torch.cat(
        [
            gray_t.squeeze(0),
            torch.from_numpy(darkness[None, :, :].astype(np.float32)),
            contrast.squeeze(0),
        ],
        dim=0,
    )
    return arr, darkness, features


def rough_content_bbox(gray: Image.Image, margin: int = 48) -> tuple[int, int, int, int]:
    arr = np.asarray(gray, dtype=np.uint8)
    dark = arr < 242
    row_score = dark.mean(axis=1)
    col_score = dark.mean(axis=0)
    rows = np.flatnonzero(row_score > 0.002)
    cols = np.flatnonzero(col_score > 0.001)
    if rows.size == 0 or cols.size == 0:
        return (0, 0, gray.width, gray.height)
    y0 = max(0, int(rows[0]) - margin)
    y1 = min(gray.height, int(rows[-1]) + margin + 1)
    x0 = max(0, int(cols[0]) - margin)
    x1 = min(gray.width, int(cols[-1]) + margin + 1)
    return (x0, y0, x1, y1)


def tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    values = list(range(0, max(1, length - tile + 1), step))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def tile_weight(height: int, width: int) -> np.ndarray:
    wy = np.hanning(max(4, height * 2))[height // 2 : height // 2 + height]
    wx = np.hanning(max(4, width * 2))[width // 2 : width // 2 + width]
    weight = np.outer(np.maximum(wy, 0.05), np.maximum(wx, 0.05)).astype(np.float32)
    return weight / max(float(weight.max()), 1e-6)


def predict_alpha(model: torch.nn.Module, features: torch.Tensor, device: torch.device, tile: int, overlap: int) -> np.ndarray:
    _, height, width = features.shape
    if max(height, width) <= tile:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(features.unsqueeze(0).to(device)).float().cpu()[0, 0]
        return pred.clamp(0, 1).numpy()

    alpha_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for y in tile_positions(height, tile, overlap):
        for x in tile_positions(width, tile, overlap):
            patch = features[:, y : y + tile, x : x + tile]
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(patch.unsqueeze(0).to(device)).float().cpu()[0, 0].clamp(0, 1).numpy()
            weight = tile_weight(pred.shape[0], pred.shape[1])
            alpha_sum[y : y + pred.shape[0], x : x + pred.shape[1]] += pred * weight
            weight_sum[y : y + pred.shape[0], x : x + pred.shape[1]] += weight
    return alpha_sum / np.maximum(weight_sum, 1e-6)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components: list[dict[str, float]] = []
    label = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x]:
                continue
            label += 1
            stack = [(x, y)]
            labels[y, x] = label
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((nx, ny))
            x_arr = np.asarray(xs, dtype=np.float32)
            y_arr = np.asarray(ys, dtype=np.float32)
            components.append(
                {
                    "label": float(label),
                    "area": float(len(xs)),
                    "x0": float(x_arr.min()),
                    "x1": float(x_arr.max()),
                    "y0": float(y_arr.min()),
                    "y1": float(y_arr.max()),
                    "var_x": float(x_arr.var()),
                    "var_y": float(y_arr.var()),
                    "cov": float(((x_arr - x_arr.mean()) * (y_arr - y_arr.mean())).mean()),
                }
            )
    return labels, components


def component_kind(component: dict[str, float], width: int, height: int, darkness: np.ndarray, args: argparse.Namespace) -> str:
    area = component["area"]
    image_area = width * height
    x0, x1 = component["x0"], component["x1"]
    y0, y1 = component["y0"], component["y1"]
    bw = x1 - x0 + 1.0
    bh = y1 - y0 + 1.0
    aspect = max(bw, bh) / max(min(bw, bh), 1.0)
    edge = x0 <= width * args.strip_edge_ratio or y0 <= height * args.strip_edge_ratio or x1 >= width * (1 - args.strip_edge_ratio) or y1 >= height * (1 - args.strip_edge_ratio)
    mean_dark = float(darkness[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1].mean())
    max_speckle = max(8.0, image_area * args.max_speckle_area_ratio)
    min_keep = max(18.0, image_area * args.min_keep_area_ratio)
    trace = component["var_x"] + component["var_y"]
    det = component["var_x"] * component["var_y"] - component["cov"] * component["cov"]
    disc = max(trace * trace / 4.0 - det, 0.0)
    major = trace / 2.0 + math.sqrt(disc)
    minor = max(trace / 2.0 - math.sqrt(disc), 1e-5)
    elongation = math.sqrt(max(major, 1e-5) / minor)
    angle = abs(math.degrees(0.5 * math.atan2(2.0 * component["cov"], component["var_x"] - component["var_y"] + 1e-9)))
    diagonal = 15.0 <= angle <= 75.0

    if edge and bw > width * 0.42 and bh < height * 0.24:
        return "strip"
    if diagonal and elongation >= args.line_aspect and area < image_area * 0.045:
        return "line"
    if aspect >= args.line_aspect * 1.6 and area < image_area * 0.04:
        return "line"
    if area < max_speckle and mean_dark < 0.38:
        return "speckle"
    if area < min_keep and mean_dark < 0.24:
        return "speckle"
    return "keep"


def overlaps_expanded(component, other, expand: float) -> bool:
    return not (
        component.x1 < other.x0 - expand
        or component.x0 > other.x1 + expand
        or component.y1 < other.y0 - expand
        or component.y0 > other.y1 + expand
    )


def near_right_of_major(component, major, x_expand: float, y_expand: float) -> bool:
    component_cx = (component.x0 + component.x1) * 0.5
    component_cy = (component.y0 + component.y1) * 0.5
    major_width = major.x1 - major.x0 + 1.0
    vertical_near = major.y0 - y_expand <= component_cy <= major.y1 + y_expand
    right_side = component_cx >= major.x1 - max(18.0, major_width * 0.18)
    not_far_right = component.x0 <= major.x1 + x_expand
    return vertical_near and right_side and not_far_right


def inside_major_dot_zone(component, major, expand: float) -> bool:
    component_cx = (component.x0 + component.x1) * 0.5
    component_cy = (component.y0 + component.y1) * 0.5
    return major.x0 - expand <= component_cx <= major.x1 + expand and major.y0 - expand <= component_cy <= major.y1 + expand


def is_big_round_matra_dot(component, width: int, height: int, major_components: list) -> bool:
    image_area = width * height
    area = component.area
    min_dot_area = max(180.0, image_area * 0.00028)
    max_dot_area = max(2600.0, image_area * 0.009)
    aspect = component.bbox_aspect
    elongation = component.elongation()
    fill = component.fill
    dark = component.mean_darkness
    x_expand = max(42.0, width * 0.075)
    y_expand = max(16.0, height * 0.035)
    near_right = any(near_right_of_major(component, major, x_expand, y_expand) for major in major_components)
    inside_major = any(inside_major_dot_zone(component, major, 12.0) for major in major_components)
    return (
        min_dot_area <= area <= max_dot_area
        and aspect <= 2.35
        and elongation <= 2.8
        and fill >= 0.22
        and dark >= 0.34
        and (near_right or inside_major)
    )


def is_valid_near_character_dot(component, width: int, height: int, major_components: list) -> bool:
    image_area = width * height
    area = component.area
    min_dot_area = max(90.0, image_area * 0.00013)
    max_dot_area = max(2600.0, image_area * 0.009)
    near_character = any(overlaps_expanded(component, major, max(58.0, width * 0.052)) for major in major_components)
    return (
        min_dot_area <= area <= max_dot_area
        and component.bbox_aspect <= 2.35
        and component.elongation() <= 2.7
        and component.fill >= 0.55
        and component.mean_darkness >= 0.255
        and near_character
    )


def rle_component_kind(component, width: int, height: int, args: argparse.Namespace, major_components: list) -> str:
    image_area = width * height
    edge = (
        component.x0 <= width * args.strip_edge_ratio
        or component.y0 <= height * args.strip_edge_ratio
        or component.x1 >= width * (1 - args.strip_edge_ratio)
        or component.y1 >= height * (1 - args.strip_edge_ratio)
    )
    max_speckle = max(8.0, image_area * args.max_speckle_area_ratio)
    min_keep = max(18.0, image_area * args.min_keep_area_ratio)
    tiny_detached = max(650.0, image_area * 0.0012)
    detached_mark_area = max(1400.0, image_area * 0.0045)
    near_major = any(overlaps_expanded(component, major, 18.0) for major in major_components)
    elongation = component.elongation()
    aspect = component.bbox_aspect
    diagonal = component.diagonal_angle()

    if edge and component.width > width * 0.34 and component.height < height * 0.18:
        return "strip"
    if edge and component.width > width * 0.16 and component.height < height * 0.14 and aspect >= 4.0:
        return "strip"
    if is_valid_near_character_dot(component, width, height, major_components):
        return "keep"
    if is_big_round_matra_dot(component, width, height, major_components):
        return "keep"
    if component.area < tiny_detached:
        return "speckle"
    if edge and component.area < detached_mark_area * 1.6:
        return "speckle"
    if component.area < detached_mark_area:
        return "speckle"
    if diagonal and elongation >= args.line_aspect and component.area < image_area * 0.045:
        return "line"
    if aspect >= args.line_aspect * 1.6 and component.area < image_area * 0.04:
        return "line"
    if component.area < max_speckle and component.mean_darkness < 0.38:
        return "speckle"
    if component.area < min_keep and component.mean_darkness < 0.24:
        return "speckle"
    return "keep"


def postprocess_alpha(alpha: np.ndarray, darkness: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    alpha = rescue_dark_round_dots(alpha, darkness, args)
    candidate = alpha >= args.alpha_threshold
    from soft_extract_foreground_v2 import connected_components as rle_connected_components

    rows, components, uf = rle_connected_components(candidate, darkness)
    keep = np.zeros_like(candidate)
    rejected = {
        "rejected_speckles": np.zeros_like(alpha, dtype=np.float32),
        "rejected_lines": np.zeros_like(alpha, dtype=np.float32),
        "rejected_strips": np.zeros_like(alpha, dtype=np.float32),
    }
    height, width = candidate.shape
    image_area = width * height
    major_area = max(5000.0, image_area * 0.009)
    major_components = [component for component in components.values() if component.area >= major_area]
    kinds = {root: rle_component_kind(component, width, height, args, major_components) for root, component in components.items()}
    for y, runs in enumerate(rows):
        for x0, x1, label in runs:
            root = uf.find(label)
            kind = kinds.get(root, "speckle")
            part = np.s_[y, x0 : x1 + 1]
            if kind == "keep":
                keep[part] = True
            else:
                rejected[f"rejected_{kind}s"][part] = alpha[part]
    for component in components.values():
        kind = kinds.get(component.root, "speckle")
        if kind == "keep":
            continue
    keep_img = to_image(keep.astype(np.float32)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.75))
    keep_soft = np.asarray(keep_img, dtype=np.float32) / 255.0
    clean_alpha = np.clip(alpha * keep_soft, 0.0, 1.0)
    clean_alpha = np.where(clean_alpha >= 0.035, clean_alpha, 0.0)
    debug = {"keep_mask": keep.astype(np.float32), **rejected}
    clean_alpha, strip_alpha = suppress_horizontal_strips(clean_alpha)
    debug["rejected_strips"] = np.maximum(debug["rejected_strips"], strip_alpha)
    debug["keep_mask"] = np.where(clean_alpha > 0.0, debug["keep_mask"], 0.0)
    return clean_alpha, debug


def rescue_dark_round_dots(alpha: np.ndarray, darkness: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from soft_extract_foreground_v2 import connected_components as rle_connected_components

    alpha_candidate = alpha >= args.alpha_threshold
    rows, components, _ = rle_connected_components(alpha_candidate, darkness)
    height, width = alpha.shape
    image_area = width * height
    major_area = max(5000.0, image_area * 0.009)
    major_components = [component for component in components.values() if component.area >= major_area]
    if not major_components:
        return alpha

    dot_mask = darkness >= 0.24
    dot_rows, dot_components, dot_uf = rle_connected_components(dot_mask, darkness)
    rescued = alpha.copy()
    min_dot_area = max(90.0, image_area * 0.00013)
    max_dot_area = max(2600.0, image_area * 0.009)
    for component in dot_components.values():
        if component.area < min_dot_area or component.area > max_dot_area:
            continue
        if component.bbox_aspect > 2.25 or component.elongation() > 2.55:
            continue
        if component.fill < 0.55 or component.mean_darkness < 0.255:
            continue
        near_word = any(overlaps_expanded(component, major, max(54.0, width * 0.048)) for major in major_components)
        valid_matra = near_word or is_big_round_matra_dot(component, width, height, major_components)
        if not valid_matra:
            continue
        for y, runs in enumerate(dot_rows):
            for x0, x1, label in runs:
                if dot_uf.find(label) == component.root:
                    rescued[y, x0 : x1 + 1] = np.maximum(
                        rescued[y, x0 : x1 + 1],
                        np.maximum(0.82, np.clip(darkness[y, x0 : x1 + 1] * 1.65, 0.0, 1.0)),
                    )
    return rescued


def mask_row_runs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    if not mask_row.any():
        return []
    padded = np.pad(mask_row.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def suppress_horizontal_strips(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = alpha > 0.05
    height, width = mask.shape
    long_run = max(90, int(width * 0.32))
    edge_long_run = max(75, int(width * 0.15))
    strip_rows = np.zeros(height, dtype=bool)
    for y in range(height):
        runs = mask_row_runs(mask[y])
        longest = max((x1 - x0 + 1 for x0, x1 in runs), default=0)
        row_near_edge = y <= height * 0.18 or y >= height * 0.82
        if longest >= long_run or (row_near_edge and longest >= edge_long_run) or mask[y].mean() >= 0.48:
            strip_rows[y] = True

    rejected = np.zeros_like(alpha, dtype=np.float32)
    cleaned = alpha.copy()
    y = 0
    max_band_h = max(18, int(height * 0.18))
    while y < height:
        if not strip_rows[y]:
            y += 1
            continue
        y0 = y
        while y < height and strip_rows[y]:
            y += 1
        y1 = y
        band_h = y1 - y0
        near_edge = y0 <= height * 0.18 or y1 >= height * 0.82
        if band_h > max_band_h and not near_edge:
            continue
        for row in range(max(0, y0 - 1), min(height, y1 + 1)):
            row_near_edge = row <= height * 0.18 or row >= height * 0.82
            row_run_cutoff = edge_long_run if row_near_edge else long_run
            for x0, x1 in mask_row_runs(mask[row]):
                length = x1 - x0 + 1
                if length >= row_run_cutoff:
                    rejected[row, x0 : x1 + 1] = np.maximum(rejected[row, x0 : x1 + 1], cleaned[row, x0 : x1 + 1])
                    cleaned[row, x0 : x1 + 1] = 0.0
    return cleaned, rejected


def compose_outputs(gray: np.ndarray, darkness: np.ndarray, alpha: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    source_soft = np.clip(gray - args.darken_strength * darkness * alpha, 0.0, 1.0)
    cleaned_soft = np.clip(1.0 - alpha + source_soft * alpha, 0.0, 1.0)
    source_ocr = np.clip(gray - args.ocr_darken * np.power(np.maximum(darkness, alpha), 0.9) * alpha, 0.0, 1.0)
    cleaned_dark = np.clip(1.0 - alpha + source_ocr * alpha, 0.0, 1.0)
    return cleaned_soft, cleaned_dark


def save_debug(out_dir: Path, stem: str, channels: dict[str, np.ndarray]) -> None:
    for name, value in channels.items():
        target = out_dir / name / f"{stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        to_image(value).save(target)


def make_preview(records: list[tuple[str, dict[str, np.ndarray]]], path: Path) -> None:
    tile = 128
    label_h = 22
    gutter = 8
    margin = 14
    rows = min(len(records), 18)
    sheet = Image.new(
        "RGB",
        (margin * 2 + len(DEBUG_CHANNELS) * tile + (len(DEBUG_CHANNELS) - 1) * gutter, margin * 2 + rows * (tile + label_h + gutter)),
        (248, 248, 244),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, (_, channels) in enumerate(records[:rows]):
        y = margin + row * (tile + label_h + gutter)
        for col, name in enumerate(DEBUG_CHANNELS):
            image = to_image(channels[name])
            image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x = margin + col * (tile + gutter)
            sheet.paste(image.convert("RGB"), (x + (tile - image.width) // 2, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=(210, 210, 205))
            draw.text((x, y + tile + 4), name, fill=(30, 30, 30), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    args = parse_args()
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

    input_dir = args.input if args.input.is_absolute() else root / args.input
    output_dir = args.output if args.output.is_absolute() else root / args.output
    if input_dir.is_file():
        images = [input_dir]
        input_root = input_dir.parent
    else:
        images = list_images(input_dir)
        input_root = input_dir
    if args.max_images:
        images = images[: args.max_images]
    records: list[tuple[str, dict[str, np.ndarray]]] = []
    for index, path in enumerate(images, start=1):
        rel = path.relative_to(input_root)
        stem = rel.with_suffix("").as_posix().replace("/", "__")
        gray_img = Image.open(path).convert("L")
        full_gray = np.asarray(gray_img, dtype=np.float32) / 255.0
        x0, y0, x1, y1 = rough_content_bbox(gray_img)
        crop_img = gray_img.crop((x0, y0, x1, y1))
        if args.work_max_side > 0 and max(crop_img.size) > args.work_max_side:
            scale = args.work_max_side / float(max(crop_img.size))
            crop_img = crop_img.resize((max(1, int(crop_img.width * scale)), max(1, int(crop_img.height * scale))), Image.Resampling.LANCZOS)
        gray, darkness, features = input_features(crop_img)
        candidate_crop = predict_alpha(model, features, device, args.tile, args.overlap)
        alpha_crop, debug_crop = postprocess_alpha(candidate_crop, darkness, args)
        cleaned_soft_crop, cleaned_dark_crop = compose_outputs(gray, darkness, alpha_crop, args)
        if args.full_canvas:
            full_shape = full_gray.shape
            candidate_raw = np.zeros(full_shape, dtype=np.float32)
            alpha = np.zeros(full_shape, dtype=np.float32)
            darkness_full = np.zeros(full_shape, dtype=np.float32)
            cleaned_soft = np.ones(full_shape, dtype=np.float32)
            cleaned_dark = np.ones(full_shape, dtype=np.float32)
            debug = {key: np.zeros(full_shape, dtype=np.float32) for key in ["keep_mask", "rejected_speckles", "rejected_lines", "rejected_strips"]}
            candidate_raw[y0:y1, x0:x1] = candidate_crop
            alpha[y0:y1, x0:x1] = alpha_crop
            darkness_full[y0:y1, x0:x1] = darkness
            cleaned_soft[y0:y1, x0:x1] = cleaned_soft_crop
            cleaned_dark[y0:y1, x0:x1] = cleaned_dark_crop
            for key, value in debug_crop.items():
                debug[key][y0:y1, x0:x1] = value
            channel_input = full_gray
            channel_darkness = darkness_full
        else:
            candidate_raw = candidate_crop
            alpha = alpha_crop
            debug = debug_crop
            cleaned_soft = cleaned_soft_crop
            cleaned_dark = cleaned_dark_crop
            channel_input = gray
            channel_darkness = darkness
        channels = {
            "input": channel_input,
            "darkness": channel_darkness,
            "candidate_raw": candidate_raw,
            **debug,
            "alpha": alpha,
            "cleaned_soft": cleaned_soft,
            "cleaned_dark_ocr": cleaned_dark,
        }
        save_debug(output_dir, stem, channels)
        records.append((stem, channels))
        print(f"[{index}/{len(images)}] {rel}")
    preview_path = args.preview if args.preview.is_absolute() else root / args.preview
    make_preview(records, preview_path)
    print(f"processed: {len(records)}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output: {output_dir}")
    print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()
