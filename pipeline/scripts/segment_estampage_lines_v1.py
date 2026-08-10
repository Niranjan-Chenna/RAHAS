from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class LineTrack:
    index: int
    xs: list[float]
    ys: list[float]
    scores: list[float]

    @property
    def count(self) -> int:
        return len(self.xs)

    @property
    def x0(self) -> float:
        return min(self.xs)

    @property
    def x1(self) -> float:
        return max(self.xs)

    @property
    def y_mean(self) -> float:
        return float(np.mean(self.ys))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic-depth curved line segmentation for estampage pages.")
    parser.add_argument("--input", type=Path, default=Path("datasets/original_estampages/negative_brahmi_estampages_page-0005.jpg"))
    parser.add_argument("--output", type=Path, default=Path("results/segmentation/line_segments/page_0005_dynamic_depth"))
    parser.add_argument("--preview", type=Path, default=Path("results/segmentation/previews/page_0005_lines_preview.png"))
    parser.add_argument("--work_max_side", type=int, default=2304)
    parser.add_argument("--window_width", type=int, default=96)
    parser.add_argument("--window_step", type=int, default=48)
    parser.add_argument("--min_line_gap", type=int, default=54)
    parser.add_argument("--peak_smooth", type=int, default=25)
    parser.add_argument("--max_link_dy", type=int, default=42)
    parser.add_argument("--band_half_height", type=int, default=44)
    parser.add_argument("--edge_band_half_height", type=int, default=78)
    parser.add_argument("--separator_margin", type=int, default=0)
    parser.add_argument("--boundary_smooth", type=int, default=92)
    parser.add_argument("--straight_line_height", type=int, default=180)
    parser.add_argument("--save_rectangular_debug", action="store_true")
    parser.add_argument("--crop_pad_x", type=int, default=42)
    parser.add_argument("--min_track_windows", type=int, default=5)
    parser.add_argument("--min_track_width_ratio", type=float, default=0.08)
    parser.add_argument("--depth_percentile", type=float, default=86.0)
    parser.add_argument("--min_depth", type=float, default=0.055)
    parser.add_argument("--faint_depth_percentile", type=float, default=80.0)
    parser.add_argument("--faint_min_depth", type=float, default=0.045)
    parser.add_argument("--strong_density_threshold", type=float, default=0.075)
    parser.add_argument("--faint_density_threshold", type=float, default=0.055)
    parser.add_argument("--interior_threshold", type=float, default=0.30)
    parser.add_argument("--interior_erode", type=int, default=25)
    parser.add_argument("--interior_boundary_margin", type=int, default=22)
    parser.add_argument("--ignore_top_ratio", type=float, default=0.035)
    parser.add_argument("--ignore_bottom_ratio", type=float, default=0.045)
    parser.add_argument("--global_peak_smooth", type=int, default=43)
    parser.add_argument("--global_peak_ratio", type=float, default=0.18)
    parser.add_argument("--line_search_radius", type=int, default=54)
    parser.add_argument("--max_curve_dy", type=int, default=94)
    parser.add_argument("--min_window_score_ratio", type=float, default=0.07)
    parser.add_argument("--dedupe_gap_ratio", type=float, default=0.36)
    parser.add_argument("--auto_zoom_line_height", type=int, default=190)
    parser.add_argument("--min_auto_zoom", type=float, default=1.75)
    parser.add_argument("--max_auto_zoom", type=float, default=4.0)
    parser.add_argument("--conversion_max_side", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers for exporting line crops. 0 chooses automatically.")
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--max_lines", type=int, default=0)
    return parser.parse_args()


def to_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L")


def stretch_depth(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    positive = depth[mask & (depth > 0.01)]
    if positive.size < 32:
        return np.clip(depth, 0.0, 1.0)
    lo = float(np.percentile(positive, 28.0))
    hi = float(np.percentile(positive, 99.2))
    if hi <= lo + 1e-5:
        return np.clip(depth, 0.0, 1.0)
    stretched = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return np.power(stretched, 0.72)


def save_converted_variants(gray: Image.Image, output_prefix: Path, max_side: int = 1600) -> None:
    work = gray
    if max_side > 0 and max(gray.size) > max_side:
        scale = max_side / float(max(gray.size))
        size = (max(1, int(round(gray.width * scale))), max(1, int(round(gray.height * scale))))
        work = gray.resize(size, Image.Resampling.BILINEAR)
    _, bg, depth = quick_relative_depth(work)
    interior = build_interior_mask(bg, argparse.Namespace(interior_threshold=0.24, interior_erode=1))
    soft = stretch_depth(depth, interior)
    soft_image = to_image(soft)
    if soft_image.size != gray.size:
        soft_image = soft_image.resize(gray.size, Image.Resampling.BILINEAR)
    soft_image.save(output_prefix.with_name(output_prefix.name + "_soft_depth.png"))
    soft_image.save(output_prefix.with_name(output_prefix.name + "_white_on_black.png"))
    soft = np.asarray(soft_image, dtype=np.float32) / 255.0
    dark_on_white = 1.0 - np.clip(soft * 0.94, 0.0, 0.94)
    to_image(dark_on_white).save(output_prefix.with_name(output_prefix.name + "_dark_on_white.png"))


def auto_zoom_line(image: Image.Image, target_height: int, min_zoom: float, max_zoom: float) -> tuple[Image.Image, float]:
    if target_height <= 0 or image.height <= 0:
        return image, 1.0
    zoom = min(max_zoom, max(min_zoom, target_height / float(image.height)))
    if zoom <= 1.01:
        return image, 1.0
    size = (max(1, int(round(image.width * zoom))), max(1, int(round(image.height * zoom))))
    return image.resize(size, Image.Resampling.LANCZOS), zoom


def resize_for_work(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0 or max(image.size) <= max_side:
        return image, 1.0
    scale = max_side / float(max(image.size))
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def local_background(gray: Image.Image) -> Image.Image:
    radius = max(31, min(121, (min(gray.size) // 18) | 1))
    return gray.filter(ImageFilter.MaxFilter(radius)).filter(ImageFilter.GaussianBlur(radius / 3.8))


def local_relative_depth(gray: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    bg = np.asarray(local_background(gray), dtype=np.float32) / 255.0
    depth = np.clip((bg - arr) / np.maximum(bg, 0.08), 0.0, 1.0)
    depth = np.power(depth, 0.82)
    return arr, bg, depth


def quick_relative_depth(gray: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    radius = max(7, min(17, (min(gray.size) // 16) | 1))
    bg_img = gray.filter(ImageFilter.MaxFilter(radius)).filter(ImageFilter.GaussianBlur(max(1.0, radius / 3.0)))
    bg = np.asarray(bg_img, dtype=np.float32) / 255.0
    depth = np.clip((bg - arr) / np.maximum(bg, 0.08), 0.0, 1.0)
    depth = np.power(depth, 0.78)
    return arr, bg, depth


def smooth_1d(values: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = np.hanning(radius * 2 + 3).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1e-6)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def build_interior_mask(background: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask = background > args.interior_threshold
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MedianFilter(7))
    erode = max(1, int(args.interior_erode) | 1)
    if erode > 1:
        mask_img = mask_img.filter(ImageFilter.MinFilter(erode))
    return np.asarray(mask_img, dtype=np.uint8) > 127


def dynamic_stroke_map(depth: np.ndarray, interior: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    strong = np.zeros_like(depth, dtype=bool)
    faint = np.zeros_like(depth, dtype=bool)
    step = max(16, args.window_step)
    win = max(step, args.window_width)
    for x0 in range(0, w, step):
        x1 = min(w, x0 + win)
        patch = depth[:, x0:x1]
        positive = patch[patch > args.faint_min_depth * 0.55]
        if positive.size < 20:
            continue
        strong_threshold = max(args.min_depth, float(np.percentile(positive, args.depth_percentile)))
        faint_threshold = max(args.faint_min_depth, float(np.percentile(positive, args.faint_depth_percentile)))
        strong[:, x0:x1] |= patch >= strong_threshold
        faint[:, x0:x1] |= patch >= faint_threshold
    strong &= interior
    faint &= interior
    # The strong map blocks page texture; the faint map rescues tiny/light characters.
    strong_density = np.asarray(to_image(strong.astype(np.float32)).filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0
    faint_density = np.asarray(to_image(faint.astype(np.float32)).filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) / 255.0
    strong_cluster = (strong_density > args.strong_density_threshold) & interior
    faint_cluster = (faint_density > args.faint_density_threshold) & interior
    support = strong_cluster | faint_cluster
    score = np.where(
        support,
        np.power(depth, 1.45) * (0.20 + strong_density * 2.2 + faint_density * 0.85),
        0.0,
    ).astype(np.float32)
    return support, score


def suppress_interior_boundaries(score: np.ndarray, interior: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    margin = max(0, int(args.interior_boundary_margin))
    if margin <= 0:
        return score
    clean = score.copy()
    h, w = score.shape
    for x in range(w):
        ys = np.flatnonzero(interior[:, x])
        if ys.size < margin * 2 + 8:
            clean[:, x] = 0.0
            continue
        top = int(ys[0])
        bottom = int(ys[-1])
        clean[: min(h, top + margin), x] = 0.0
        clean[max(0, bottom - margin + 1) :, x] = 0.0
    return clean


def column_windows(width: int, args: argparse.Namespace) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    for x0 in range(0, max(1, width - args.window_width + 1), args.window_step):
        x1 = min(width, x0 + args.window_width)
        windows.append((x0, x1, (x0 + x1) // 2))
    if not windows or windows[-1][1] < width:
        x1 = width
        x0 = max(0, width - args.window_width)
        windows.append((x0, x1, (x0 + x1) // 2))
    return windows


def find_peaks(profile: np.ndarray, min_gap: int) -> list[tuple[int, float]]:
    if profile.max() <= 0:
        return []
    threshold = max(float(profile.mean() + profile.std() * 0.35), float(profile.max() * 0.16))
    raw: list[tuple[int, float]] = []
    for y in range(1, len(profile) - 1):
        if profile[y] >= threshold and profile[y] >= profile[y - 1] and profile[y] >= profile[y + 1]:
            raw.append((y, float(profile[y])))
    raw.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for y, score in raw:
        if all(abs(y - existing_y) >= min_gap for existing_y, _ in selected):
            selected.append((y, score))
    return sorted(selected)


def find_ranked_peaks(profile: np.ndarray, min_gap: int, threshold_ratio: float) -> list[tuple[int, float]]:
    if profile.max() <= 0:
        return []
    threshold = float(profile.max()) * threshold_ratio
    raw: list[tuple[int, float]] = []
    for y in range(1, len(profile) - 1):
        if profile[y] >= threshold and profile[y] >= profile[y - 1] and profile[y] >= profile[y + 1]:
            raw.append((y, float(profile[y])))
    raw.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for y, score in raw:
        if all(abs(y - existing_y) >= min_gap for existing_y, _ in selected):
            selected.append((y, score))
    return sorted(selected)


def window_peaks(score: np.ndarray, args: argparse.Namespace) -> list[tuple[int, list[tuple[int, float]]]]:
    _, w = score.shape
    peaks_by_x: list[tuple[int, list[tuple[int, float]]]] = []
    for x0, x1, xc in column_windows(w, args):
        profile = score[:, x0:x1].sum(axis=1)
        profile = smooth_1d(profile, args.peak_smooth)
        peaks = find_peaks(profile, args.min_line_gap)
        peaks_by_x.append((xc, peaks))
    return peaks_by_x


def global_line_seeds(score: np.ndarray, interior: np.ndarray, args: argparse.Namespace) -> list[tuple[int, float]]:
    h, _ = score.shape
    width_profile = np.maximum(interior.sum(axis=1).astype(np.float32), 1.0)
    profile = score.sum(axis=1) / np.sqrt(width_profile)
    profile = smooth_1d(profile, args.global_peak_smooth)
    top = int(round(h * args.ignore_top_ratio))
    bottom = int(round(h * (1.0 - args.ignore_bottom_ratio)))
    profile[:top] = 0.0
    profile[bottom:] = 0.0
    peaks = find_ranked_peaks(profile, args.min_line_gap, args.global_peak_ratio)
    if not peaks:
        return []
    return sorted(peaks)


def trace_seeded_tracks(
    seeds: list[tuple[int, float]],
    score: np.ndarray,
    interior: np.ndarray,
    args: argparse.Namespace,
) -> list[LineTrack]:
    h, w = score.shape
    windows = column_windows(w, args)
    tracks: list[LineTrack] = []
    top = h * args.ignore_top_ratio
    bottom = h * (1.0 - args.ignore_bottom_ratio)
    for index, (seed_y, seed_score) in enumerate(seeds, start=1):
        xs: list[float] = []
        ys: list[float] = []
        scores: list[float] = []
        previous_y = float(seed_y)
        previous_delta = 0.0
        for x0, x1, xc in windows:
            interior_width = interior[:, x0:x1].sum(axis=1).astype(np.float32)
            if float(interior_width.max()) < max(8.0, (x1 - x0) * 0.14):
                continue
            profile = score[:, x0:x1].sum(axis=1) / np.sqrt(np.maximum(interior_width, 1.0))
            profile = smooth_1d(profile, args.peak_smooth)
            local_max = float(profile.max())
            if local_max <= 0.0:
                continue
            predicted = previous_y + previous_delta
            predicted = float(np.clip(predicted, seed_y - args.max_curve_dy, seed_y + args.max_curve_dy))
            y0 = max(0, int(round(predicted - args.line_search_radius)))
            y1 = min(h, int(round(predicted + args.line_search_radius + 1)))
            if y1 <= y0:
                continue
            local_slice = profile[y0:y1]
            best_y = int(y0 + int(local_slice.argmax()))
            best_score = float(profile[best_y])
            if best_score < max(local_max * args.min_window_score_ratio, seed_score * 0.035):
                continue
            if not (top <= best_y <= bottom):
                continue
            xs.append(float(xc))
            ys.append(float(best_y))
            scores.append(best_score)
            if len(ys) >= 2:
                previous_delta = float(np.clip(ys[-1] - ys[-2], -args.line_search_radius * 0.45, args.line_search_radius * 0.45))
            previous_y = float(best_y)

        if not xs:
            continue
        tracks.append(LineTrack(index=index, xs=xs, ys=ys, scores=scores))

    min_width = w * args.min_track_width_ratio
    tracks = [track for track in tracks if track.count >= args.min_track_windows and (track.x1 - track.x0) >= min_width]
    tracks = dedupe_tracks(tracks, args)
    tracks.sort(key=lambda track: track.y_mean)
    if args.max_lines:
        tracks = tracks[: args.max_lines]
    for i, track in enumerate(tracks, start=1):
        track.index = i
    return tracks


def dedupe_tracks(tracks: list[LineTrack], args: argparse.Namespace) -> list[LineTrack]:
    kept: list[LineTrack] = []
    for track in sorted(tracks, key=lambda item: float(np.median(item.scores)) if item.scores else 0.0, reverse=True):
        if all(abs(track.y_mean - existing.y_mean) >= args.min_line_gap * args.dedupe_gap_ratio for existing in kept):
            kept.append(track)
    return kept


def link_tracks(peaks_by_x: list[tuple[int, list[tuple[int, float]]]], args: argparse.Namespace, width: int) -> list[LineTrack]:
    tracks: list[LineTrack] = []
    next_index = 1
    active: list[LineTrack] = []
    for xc, peaks in peaks_by_x:
        used: set[int] = set()
        for track in list(active):
            prediction = track.ys[-1]
            if len(track.ys) >= 3:
                prediction = track.ys[-1] + (track.ys[-1] - track.ys[-3]) / max(1, min(3, len(track.ys) - 1))
            best_i = -1
            best_cost = 1e9
            for i, (py, score) in enumerate(peaks):
                if i in used:
                    continue
                cost = abs(py - prediction)
                if cost < best_cost and cost <= args.max_link_dy:
                    best_i = i
                    best_cost = cost
            if best_i >= 0:
                py, score = peaks[best_i]
                track.xs.append(float(xc))
                track.ys.append(float(py))
                track.scores.append(float(score))
                used.add(best_i)
        for i, (py, score) in enumerate(peaks):
            if i in used:
                continue
            track = LineTrack(index=next_index, xs=[float(xc)], ys=[float(py)], scores=[float(score)])
            next_index += 1
            tracks.append(track)
            active.append(track)
        active = [track for track in active if track.xs[-1] >= xc - args.window_step * 3]

    min_width = width * args.min_track_width_ratio
    tracks = [track for track in tracks if track.count >= args.min_track_windows and (track.x1 - track.x0) >= min_width]
    tracks.sort(key=lambda track: track.y_mean)
    if args.max_lines:
        tracks = tracks[: args.max_lines]
    for i, track in enumerate(tracks, start=1):
        track.index = i
    return tracks


def track_y_at(track: LineTrack, x_values: np.ndarray, width: int) -> np.ndarray:
    xs = np.asarray(track.xs, dtype=np.float32)
    ys = np.asarray(track.ys, dtype=np.float32)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if xs.size < 2:
        return np.full_like(x_values, ys[0] if ys.size else 0.0, dtype=np.float32)
    interp = np.interp(x_values, xs, ys, left=np.nan, right=np.nan).astype(np.float32)
    valid = ~np.isnan(interp)
    if valid.any():
        first = np.flatnonzero(valid)[0]
        last = np.flatnonzero(valid)[-1]
        interp[:first] = interp[first]
        interp[last + 1 :] = interp[last]
    interp = smooth_1d(interp, 10)
    return interp


def save_line_crop(
    image: Image.Image,
    source_original: Image.Image,
    depth: np.ndarray,
    stroke: np.ndarray,
    track: LineTrack,
    previous_track: LineTrack | None,
    next_track: LineTrack | None,
    output: Path,
    scale_to_original: float,
    args: argparse.Namespace,
) -> dict[str, str | int | float]:
    w, h = image.size
    x_values = np.arange(w, dtype=np.float32)
    y_center = smooth_1d(track_y_at(track, x_values, w), args.boundary_smooth)
    upper = y_center - args.edge_band_half_height
    lower = y_center + args.edge_band_half_height
    if previous_track is not None:
        previous_y = smooth_1d(track_y_at(previous_track, x_values, w), args.boundary_smooth)
        upper = np.maximum(upper, ((previous_y + y_center) / 2.0) + args.separator_margin)
    if next_track is not None:
        next_y = smooth_1d(track_y_at(next_track, x_values, w), args.boundary_smooth)
        lower = np.minimum(lower, ((next_y + y_center) / 2.0) - args.separator_margin)
    upper = smooth_1d(upper, max(4, args.boundary_smooth // 2))
    lower = smooth_1d(lower, max(4, args.boundary_smooth // 2))
    upper = np.clip(upper, 0, h - 1)
    lower = np.clip(np.maximum(lower, upper + 4), 0, h)

    x0 = max(0, int(math.floor(track.x0 - args.crop_pad_x)))
    x1 = min(w, int(math.ceil(track.x1 + args.crop_pad_x)))
    y0 = max(0, int(math.floor(np.nanmin(upper[x0:x1]))))
    y1 = min(h, int(math.ceil(np.nanmax(lower[x0:x1]))))
    line_dir = output / f"line_{track.index:02d}"
    line_dir.mkdir(parents=True, exist_ok=True)
    raw_rect = image.crop((x0, y0, x1, y1))
    mask_arr = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for local_x, source_x in enumerate(range(x0, x1)):
        top = int(max(0, math.floor(upper[source_x] - y0)))
        bottom = int(min(y1 - y0, math.ceil(lower[source_x] - y0)))
        if bottom > top:
            mask_arr[top:bottom, local_x] = 1.0
    mask = to_image(mask_arr).filter(ImageFilter.GaussianBlur(0.8))
    white = Image.new("L", raw_rect.size, 255)
    raw = Image.composite(raw_rect, white, mask)
    raw.save(line_dir / "input.png")
    if args.save_rectangular_debug:
        raw_rect.save(line_dir / "input_rectangular_debug.png")
    depth_crop = depth[y0:y1, x0:x1] * mask_arr
    stroke_crop = stroke[y0:y1, x0:x1].astype(np.float32) * mask_arr
    to_image(depth_crop).save(line_dir / "relative_depth.png")
    to_image(stroke_crop).save(line_dir / "stroke_candidate.png")
    mask.save(line_dir / "line_band_mask.png")
    save_converted_variants(raw, line_dir / "converted", args.conversion_max_side)

    overlay = raw_rect.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    points: list[tuple[int, int]] = []
    upper_points: list[tuple[int, int]] = []
    lower_points: list[tuple[int, int]] = []
    for x in range(x0, x1, max(1, (x1 - x0) // 160)):
        points.append((x - x0, int(round(y_center[x] - y0))))
        upper_points.append((x - x0, int(round(upper[x] - y0))))
        lower_points.append((x - x0, int(round(lower[x] - y0))))
    if len(points) >= 2:
        draw.line(points, fill=(255, 0, 0), width=2)
        draw.line(upper_points, fill=(0, 160, 255), width=2)
        draw.line(lower_points, fill=(0, 160, 255), width=2)
    overlay.save(line_dir / "overlay.png")
    raw.convert("RGB").save(line_dir / "input_segmented_preview.png")
    straight = straighten_band(image, x0, x1, upper, lower, args.straight_line_height)
    straight.save(line_dir / "straight_input.png")
    save_converted_variants(straight, line_dir / "straight_converted", args.conversion_max_side)

    inv = 1.0 / max(scale_to_original, 1e-6)
    ox0 = max(0, int(math.floor(x0 * inv)))
    oy0 = max(0, int(math.floor(y0 * inv)))
    ox1 = min(source_original.width, int(math.ceil(x1 * inv)))
    oy1 = min(source_original.height, int(math.ceil(y1 * inv)))
    original_rect = source_original.crop((ox0, oy0, ox1, oy1))
    original_mask = mask.resize(original_rect.size, Image.Resampling.BILINEAR)
    original_white = Image.new("L", original_rect.size, 255)
    original_raw = Image.composite(original_rect, original_white, original_mask)
    original_raw.save(line_dir / "input_original_resolution.png")
    if args.save_rectangular_debug:
        original_rect.save(line_dir / "input_original_rectangular_debug.png")
    zoomed, zoom = auto_zoom_line(original_raw, args.auto_zoom_line_height, args.min_auto_zoom, args.max_auto_zoom)
    zoomed.save(line_dir / "input_auto_zoom.png")
    save_converted_variants(zoomed, line_dir / "auto_zoom_converted", args.conversion_max_side)

    return {
        "line": track.index,
        "x0_work": x0,
        "y0_work": y0,
        "x1_work": x1,
        "y1_work": y1,
        "x0_original": ox0,
        "y0_original": oy0,
        "x1_original": ox1,
        "y1_original": oy1,
        "points": track.count,
        "width": x1 - x0,
        "height": y1 - y0,
        "auto_zoom": round(zoom, 3),
    }


def straighten_band(image: Image.Image, x0: int, x1: int, upper: np.ndarray, lower: np.ndarray, height: int) -> Image.Image:
    arr = np.asarray(image, dtype=np.uint8)
    src_h, _ = arr.shape
    out_w = max(1, x1 - x0)
    out_h = max(16, int(height))
    out = np.full((out_h, out_w), 255, dtype=np.uint8)
    ratios = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
    for ox, sx in enumerate(range(x0, x1)):
        top = float(upper[sx])
        bottom = float(lower[sx])
        if bottom <= top + 1:
            continue
        ys = np.round(top + ratios * (bottom - top)).astype(np.int32)
        valid = (ys >= 0) & (ys < src_h)
        out[valid, ox] = arr[ys[valid], sx]
    return Image.fromarray(out, mode="L")


def save_preview(image: Image.Image, depth: np.ndarray, stroke: np.ndarray, tracks: list[LineTrack], output: Path, preview: Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    colors = [(255, 0, 0), (0, 128, 255), (255, 128, 0), (180, 0, 255), (0, 180, 80)]
    w, _ = image.size
    x_values = np.arange(w, dtype=np.float32)
    for track in tracks:
        color = colors[(track.index - 1) % len(colors)]
        y_center = track_y_at(track, x_values, w)
        x0 = max(0, int(track.x0))
        x1 = min(w - 1, int(track.x1))
        points = [(x, int(round(y_center[x]))) for x in range(x0, x1 + 1, max(1, (x1 - x0) // 220))]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
            draw.text((points[0][0] + 4, points[0][1] - 12), str(track.index), fill=color, font=font)

    output.mkdir(parents=True, exist_ok=True)
    overlay.save(output / "line_overlay.png")
    to_image(depth).save(output / "relative_depth.png")
    to_image(stroke.astype(np.float32)).save(output / "stroke_candidate.png")

    thumb_w = 560
    tiles = [
        ("input+lines", overlay),
        ("relative_depth", to_image(depth).convert("RGB")),
        ("stroke_candidate", to_image(stroke.astype(np.float32)).convert("RGB")),
    ]
    thumbs = []
    for _, tile in tiles:
        t = tile.copy()
        t.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
        thumbs.append(t)
    label_h = 24
    gutter = 12
    margin = 14
    sheet = Image.new("RGB", (margin * 2 + len(thumbs) * thumb_w + (len(thumbs) - 1) * gutter, margin * 2 + max(t.height for t in thumbs) + label_h), (248, 248, 244))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, ((label, _), t) in enumerate(zip(tiles, thumbs)):
        x = margin + i * (thumb_w + gutter)
        y = margin
        sheet.paste(t, (x + (thumb_w - t.width) // 2, y))
        d.rectangle((x, y, x + thumb_w, y + max(t.height for t in thumbs)), outline=(210, 210, 205))
        d.text((x, y + max(t.height for t in thumbs) + 5), label, fill=(30, 30, 30), font=font)
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview)


def make_line_contact_sheet(output: Path, tracks: list[LineTrack]) -> None:
    rows = []
    for track in tracks:
        path = output / f"line_{track.index:02d}" / "overlay.png"
        if path.exists():
            rows.append((track.index, Image.open(path).convert("RGB")))
    if not rows:
        return
    tile_w = 900
    tile_h = 120
    label_h = 22
    gutter = 8
    margin = 14
    sheet = Image.new("RGB", (margin * 2 + tile_w, margin * 2 + len(rows) * (tile_h + label_h + gutter)), (248, 248, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, (index, image) in enumerate(rows):
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = margin
        y = margin + row * (tile_h + label_h + gutter)
        sheet.paste(image, (x + (tile_w - image.width) // 2, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(210, 210, 205))
        draw.text((x, y + tile_h + 4), f"line_{index:02d}", fill=(30, 30, 30), font=font)
    sheet.save(output / "line_crops_preview.png")


def make_converted_contact_sheet(output: Path, tracks: list[LineTrack]) -> None:
    rows = []
    for track in tracks:
        path = output / f"line_{track.index:02d}" / "auto_zoom_converted_white_on_black.png"
        if path.exists():
            rows.append((track.index, Image.open(path).convert("RGB")))
    if not rows:
        return
    tile_w = 1200
    tile_h = 170
    label_h = 22
    gutter = 8
    margin = 14
    sheet = Image.new("RGB", (margin * 2 + tile_w, margin * 2 + len(rows) * (tile_h + label_h + gutter)), (248, 248, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, (index, image) in enumerate(rows):
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = margin
        y = margin + row * (tile_h + label_h + gutter)
        sheet.paste(image, (x + (tile_w - image.width) // 2, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(210, 210, 205))
        draw.text((x, y + tile_h + 4), f"line_{index:02d}_auto_zoom", fill=(30, 30, 30), font=font)
    sheet.save(output / "auto_zoom_converted_preview.png")


def make_straight_contact_sheet(output: Path, tracks: list[LineTrack]) -> None:
    rows = []
    for track in tracks:
        path = output / f"line_{track.index:02d}" / "straight_converted_white_on_black.png"
        if path.exists():
            rows.append((track.index, Image.open(path).convert("RGB")))
    if not rows:
        return
    tile_w = 1200
    tile_h = 150
    label_h = 22
    gutter = 8
    margin = 14
    sheet = Image.new("RGB", (margin * 2 + tile_w, margin * 2 + len(rows) * (tile_h + label_h + gutter)), (248, 248, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, (index, image) in enumerate(rows):
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = margin
        y = margin + row * (tile_h + label_h + gutter)
        sheet.paste(image, (x + (tile_w - image.width) // 2, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(210, 210, 205))
        draw.text((x, y + tile_h + 4), f"line_{index:02d}_straight", fill=(30, 30, 30), font=font)
    sheet.save(output / "straight_converted_preview.png")


def worker_count(args: argparse.Namespace, track_count: int) -> int:
    if track_count <= 1:
        return 1
    if args.workers and args.workers > 0:
        return max(1, min(int(args.workers), track_count))
    cpu_count = os.cpu_count() or 4
    return max(1, min(track_count, max(2, cpu_count // 2)))


def export_line_crops(
    work: Image.Image,
    source: Image.Image,
    depth: np.ndarray,
    stroke: np.ndarray,
    tracks: list[LineTrack],
    output: Path,
    scale: float,
    args: argparse.Namespace,
) -> list[dict[str, str | int | float]]:
    if not tracks:
        return []
    workers = worker_count(args, len(tracks))
    print(f"export workers: {workers}")
    if workers == 1:
        rows = []
        for i, track in enumerate(tracks):
            previous_track = tracks[i - 1] if i > 0 else None
            next_track = tracks[i + 1] if i + 1 < len(tracks) else None
            rows.append(save_line_crop(work, source, depth, stroke, track, previous_track, next_track, output, scale, args))
        return rows

    rows: list[dict[str, str | int | float]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, track in enumerate(tracks):
            previous_track = tracks[i - 1] if i > 0 else None
            next_track = tracks[i + 1] if i + 1 < len(tracks) else None
            futures.append(executor.submit(save_line_crop, work, source, depth, stroke, track, previous_track, next_track, output, scale, args))
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"exported line {row['line']} ({completed}/{len(futures)})")
    rows.sort(key=lambda item: int(item["line"]))
    return rows


def main() -> None:
    args = parse_args()
    source = Image.open(args.input).convert("L")
    work, scale = resize_for_work(source, args.work_max_side)
    _, bg, depth = local_relative_depth(work)
    interior = build_interior_mask(bg, args)
    stroke, score = dynamic_stroke_map(depth, interior, args)
    score = suppress_interior_boundaries(score, interior, args)
    seeds = global_line_seeds(score, interior, args)
    tracks = trace_seeded_tracks(seeds, score, interior, args)
    args.output.mkdir(parents=True, exist_ok=True)
    to_image(interior.astype(np.float32)).save(args.output / "stone_interior_mask.png")
    to_image(stretch_depth(depth, interior)).save(args.output / "dynamic_white_on_black.png")
    to_image(1.0 - np.clip(stretch_depth(depth, interior) * 0.94, 0.0, 0.94)).save(args.output / "dynamic_dark_on_white.png")
    save_preview(work, depth, stroke, tracks, args.output, args.preview)
    if args.preview_only:
        print(f"input: {args.input}")
        print(f"work size: {work.size[0]}x{work.size[1]} scale={scale:.4f}")
        print(f"seeds: {len(seeds)}")
        print(f"lines: {len(tracks)}")
        print(f"output: {args.output}")
        print(f"preview: {args.preview}")
        return
    rows = export_line_crops(work, source, depth, stroke, tracks, args.output, scale, args)
    manifest = args.output / "line_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    make_line_contact_sheet(args.output, tracks)
    make_converted_contact_sheet(args.output, tracks)
    make_straight_contact_sheet(args.output, tracks)
    print(f"input: {args.input}")
    print(f"work size: {work.size[0]}x{work.size[1]} scale={scale:.4f}")
    print(f"seeds: {len(seeds)}")
    print(f"lines: {len(tracks)}")
    print(f"output: {args.output}")
    print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
