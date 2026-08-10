from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from restore_sparse_stroke_v1 import connected_components, estimate_background, resize_for_work


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class RestorationProfile:
    input_path: Path
    work_width: int
    work_height: int
    scale: float
    strong_threshold: float
    stroke_pixel_ratio: float
    total_components: int
    small_noise_count: int
    small_noise_per_mpix: float
    large_component_count: int
    median_large_height: float
    median_large_area: float
    bright_background_ratio: float
    dark_ink_ratio: float
    noise_intensity: float
    method: str
    sparse_candidate_percentile: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noise-aware adaptive estampage restoration selector.")
    parser.add_argument("--input", type=Path, default=Path("datasets/original_estampages"))
    parser.add_argument("--output", type=Path, default=Path("results/restoration/adaptive_selector_v1"))
    parser.add_argument("--preview", type=Path, default=Path("results/restoration/previews/adaptive_selector_v1"))
    parser.add_argument("--profile_max_side", type=int, default=1024)
    parser.add_argument("--dense_work_max_side", type=int, default=2304)
    parser.add_argument("--background_radius", type=int, default=91)
    parser.add_argument("--profile_percentile", type=float, default=86.0)
    parser.add_argument("--min_depth", type=float, default=0.040)
    parser.add_argument("--weak_depth", type=float, default=0.020)
    parser.add_argument("--e_matra_area", type=int, default=18)
    parser.add_argument("--small_dot_max_dim", type=int, default=8)
    parser.add_argument("--noise_low_per_mpix", type=float, default=900.0)
    parser.add_argument("--noise_high_per_mpix", type=float, default=1900.0)
    parser.add_argument("--sparse_large_component_limit", type=int, default=1600)
    parser.add_argument("--sparse_min_median_height", type=float, default=12.0)
    parser.add_argument("--sparse_max_stroke_ratio", type=float, default=0.20)
    parser.add_argument("--force_method", choices=["auto", "sparse_stroke", "letter_recall"], default="auto")
    parser.add_argument("--profile_only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images = [item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_EXTENSIONS]
    return images


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def profile_image(path: Path, args: argparse.Namespace) -> RestorationProfile:
    source = Image.open(path).convert("L")
    work, scale = resize_for_work(source, args.profile_max_side)
    gray = np.asarray(work, dtype=np.float32) / 255.0
    background = estimate_background(work, args.background_radius)
    bg = np.asarray(background, dtype=np.float32) / 255.0
    depth = np.clip((bg - gray) / np.maximum(bg, 0.08), 0.0, 1.0)
    depth = np.power(depth, 0.82)
    bright_background_ratio = float((gray > 0.75).mean())
    dark_ink_ratio = float((gray < 0.35).mean())

    positive = depth[depth > args.weak_depth]
    threshold = max(args.min_depth, float(np.percentile(positive, args.profile_percentile)) if positive.size else args.min_depth)
    candidate = depth >= threshold
    alpha = np.clip(depth / max(threshold, 1e-4), 0.0, 1.0)
    comps, _ = connected_components(candidate, depth, alpha)

    scale_area = max(scale * scale, 1e-6)
    matra_area = max(4, int(round(args.e_matra_area * scale_area)))
    small_dim = max(3, int(round(args.small_dot_max_dim * max(scale, 0.4))))
    small_noise = [
        comp
        for comp in comps
        if comp.area < matra_area
        and max(comp.width, comp.height) <= small_dim
        and comp.mean_depth >= args.min_depth * 0.45
    ]
    large = [
        comp
        for comp in comps
        if comp.area >= matra_area
        or max(comp.width, comp.height) >= max(small_dim + 2, 9)
        or comp.mean_depth >= args.min_depth * 2.0
    ]
    mpix = max(1e-6, (work.width * work.height) / 1_000_000.0)
    small_noise_per_mpix = len(small_noise) / mpix
    noise_intensity = np.clip(
        (small_noise_per_mpix - args.noise_low_per_mpix)
        / max(1e-6, args.noise_high_per_mpix - args.noise_low_per_mpix),
        0.0,
        1.0,
    )

    median_large_height = float(np.median([comp.height for comp in large])) if large else 0.0
    median_large_area = float(np.median([comp.area for comp in large])) if large else 0.0
    stroke_ratio = float(candidate.mean())
    sparse_clean_paper = bright_background_ratio >= 0.68 and dark_ink_ratio <= 0.14
    sparse_connected = (
        len(large) <= args.sparse_large_component_limit
        and median_large_height >= args.sparse_min_median_height
        and stroke_ratio <= args.sparse_max_stroke_ratio
    )
    sparse_like = sparse_clean_paper or sparse_connected
    if args.force_method == "auto":
        method = "sparse_stroke" if sparse_like else "letter_recall"
    else:
        method = args.force_method

    sparse_percentile = 82.0 + 7.0 * float(noise_intensity)
    if method == "sparse_stroke" and small_noise_per_mpix > args.noise_high_per_mpix * 1.35:
        sparse_percentile = min(92.0, sparse_percentile + 2.0)

    return RestorationProfile(
        input_path=path,
        work_width=work.width,
        work_height=work.height,
        scale=scale,
        strong_threshold=threshold,
        stroke_pixel_ratio=stroke_ratio,
        total_components=len(comps),
        small_noise_count=len(small_noise),
        small_noise_per_mpix=small_noise_per_mpix,
        large_component_count=len(large),
        median_large_height=median_large_height,
        median_large_area=median_large_area,
        bright_background_ratio=bright_background_ratio,
        dark_ink_ratio=dark_ink_ratio,
        noise_intensity=float(noise_intensity),
        method=method,
        sparse_candidate_percentile=float(sparse_percentile),
    )


def write_profile(profile: RestorationProfile, output_dir: Path, command: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selector_profile.txt").open("w", encoding="utf-8") as file:
        file.write(f"input={profile.input_path}\n")
        file.write(f"method={profile.method}\n")
        file.write(f"work_size={profile.work_width}x{profile.work_height}\n")
        file.write(f"scale={profile.scale:.6f}\n")
        file.write(f"strong_threshold={profile.strong_threshold:.6f}\n")
        file.write(f"stroke_pixel_ratio={profile.stroke_pixel_ratio:.6f}\n")
        file.write(f"total_components={profile.total_components}\n")
        file.write(f"small_noise_count={profile.small_noise_count}\n")
        file.write(f"small_noise_per_mpix={profile.small_noise_per_mpix:.3f}\n")
        file.write(f"large_component_count={profile.large_component_count}\n")
        file.write(f"median_large_height={profile.median_large_height:.3f}\n")
        file.write(f"median_large_area={profile.median_large_area:.3f}\n")
        file.write(f"bright_background_ratio={profile.bright_background_ratio:.6f}\n")
        file.write(f"dark_ink_ratio={profile.dark_ink_ratio:.6f}\n")
        file.write(f"noise_intensity={profile.noise_intensity:.6f}\n")
        file.write(f"sparse_candidate_percentile={profile.sparse_candidate_percentile:.3f}\n")
        file.write("command=" + " ".join(command) + "\n")


def build_command(profile: RestorationProfile, output_dir: Path, preview_path: Path, args: argparse.Namespace) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    if profile.method == "sparse_stroke":
        return [
            sys.executable,
            str(script_dir / "restore_sparse_stroke_v1.py"),
            "--input",
            str(profile.input_path),
            "--output",
            str(output_dir),
            "--preview",
            str(preview_path),
            "--work_max_side",
            "0",
            "--background_radius",
            "121",
            "--candidate_percentile",
            f"{profile.sparse_candidate_percentile:.2f}",
            "--min_depth",
            "0.040",
            "--weak_depth",
            "0.020",
            "--matra_dot_area",
            "28",
            "--tiny_area",
            "8",
            "--connect_iters",
            "1",
            "--soft_gain",
            "2.0",
            "--dark_gain",
            "2.7",
        ]

    return [
        sys.executable,
        str(script_dir / "segment_estampage_characters_v1.py"),
        "--input",
        str(profile.input_path),
        "--output",
        str(output_dir),
        "--preview",
        str(preview_path),
        "--work_max_side",
        str(args.dense_work_max_side),
        "--max_chars",
        "240",
        "--device",
        "auto",
        "--candidate_mode",
        "recall",
        "--post_filter_mode",
        "letter_recall",
        "--adaptive_noise_intensity",
        "--alpha_threshold",
        "0.10",
        "--stroke_alpha_floor",
        "0.01",
        "--min_alpha_keep",
        "0.040",
        "--final_alpha_floor",
        "0.01",
        "--min_main_area",
        "8",
        "--min_dot_area",
        "6",
        "--matra_dot_area",
        str(args.e_matra_area),
        "--matra_dot_depth_ratio",
        "0.46",
        "--min_depth",
        "0.040",
        "--min_dot_depth",
        "0.052",
    ]


def run_one(path: Path, args: argparse.Namespace) -> RestorationProfile:
    profile = profile_image(path, args)
    stem = safe_stem(path)
    output_dir = args.output / stem
    preview_path = args.preview / f"{stem}.png"
    command = build_command(profile, output_dir, preview_path, args)
    write_profile(profile, output_dir, command)
    print(
        f"{path.name}: method={profile.method} small_noise/mpix={profile.small_noise_per_mpix:.1f} "
        f"noise={profile.noise_intensity:.3f} large={profile.large_component_count} "
        f"median_h={profile.median_large_height:.1f} bright={profile.bright_background_ratio:.3f} "
        f"dark={profile.dark_ink_ratio:.3f}"
    )
    if not args.profile_only:
        subprocess.run(command, check=True)
        write_profile(profile, output_dir, command)
    return profile


def main() -> None:
    args = parse_args()
    inputs = iter_inputs(args.input)
    if args.limit:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit(f"No images found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.preview.mkdir(parents=True, exist_ok=True)

    profiles = [run_one(path, args) for path in inputs]
    summary_path = args.output / "adaptive_summary.csv"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(
            "input,method,small_noise_per_mpix,noise_intensity,stroke_pixel_ratio,"
            "large_component_count,median_large_height,bright_background_ratio,dark_ink_ratio,"
            "sparse_candidate_percentile\n"
        )
        for profile in profiles:
            file.write(
                f"{profile.input_path.name},{profile.method},{profile.small_noise_per_mpix:.3f},"
                f"{profile.noise_intensity:.6f},{profile.stroke_pixel_ratio:.6f},"
                f"{profile.large_component_count},{profile.median_large_height:.3f},"
                f"{profile.bright_background_ratio:.6f},{profile.dark_ink_ratio:.6f},"
                f"{profile.sparse_candidate_percentile:.3f}\n"
            )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
