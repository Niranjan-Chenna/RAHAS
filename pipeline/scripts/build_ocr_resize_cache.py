from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.soft_data import fit_crop_onto_canvas


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache OCR references at model resolution without binarization.")
    parser.add_argument("--source", type=Path, default=Path("datasets/character_references"))
    parser.add_argument("--output", type=Path, default=Path("datasets/prepared/12_ocr_soft_resized_v1/characters"))
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def convert_one(job: tuple[str, str, int]) -> dict[str, object]:
    source_text, output_text, size = job
    source = Path(source_text)
    output = Path(output_text)
    if output.exists():
        return {"source": source_text, "output": output_text, "status": "cached"}
    with Image.open(source) as image:
        original_size = image.size
        canvas = fit_crop_onto_canvas(image.convert("L"), size, rng=None, train=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=1)
    return {
        "source": source_text,
        "output": output_text,
        "source_width": original_size[0],
        "source_height": original_size[1],
        "status": "written",
    }


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    jobs = []
    for source in files:
        relative = source.relative_to(args.source).with_suffix(".png")
        jobs.append((str(source.resolve()), str((args.output / relative).resolve()), args.size))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output.parent / "resize_manifest.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(convert_one, jobs, chunksize=4), start=1):
            rows.append(result)
            if index % 100 == 0 or index == len(jobs):
                print(f"CACHE {index}/{len(jobs)}", flush=True)
    fieldnames = ["source", "output", "source_width", "source_height", "status"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"complete files={len(rows)} output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
