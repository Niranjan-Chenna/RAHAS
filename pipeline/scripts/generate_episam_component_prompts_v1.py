from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Component:
    label: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    cx: float
    cy: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EpiSAM centroid/box prompts from a restoration mask.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--original-image", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--min-height", type=int, default=28)
    parser.add_argument("--max-height", type=int, default=150)
    parser.add_argument("--max-width", type=int, default=180)
    parser.add_argument("--line-step", type=int, default=105)
    parser.add_argument("--attach-fragments", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restored = np.asarray(Image.open(args.input_dir / "restored_dark_ocr.png").convert("L"), dtype=np.uint8)
    mask_path = args.input_dir / "final_keep_mask.png"
    if not mask_path.exists():
        mask_path = args.input_dir / "keep_mask.png"
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary = (mask >= 96).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components: list[Component] = []
    small: list[Component] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        component = Component(label, x, y, x + width, y + height, area, *centroids[label])
        crop_darkness = 1.0 - restored[y : y + height, x : x + width].astype(np.float32) / 255.0
        mean_ink = float(crop_darkness[binary[y : y + height, x : x + width] > 0].mean()) if area else 0.0
        if area >= args.min_area and height >= args.min_height and height <= args.max_height and width <= args.max_width and mean_ink >= 0.16:
            components.append(component)
        elif area >= 5 and max(width, height) <= 30:
            small.append(component)

    if args.attach_fragments:
        # Include nearby detached dots in prompt context without allowing transitive expansion.
        for component in components:
            original_x0, original_y0 = component.x0, component.y0
            original_x1, original_y1 = component.x1, component.y1
            expand_x = max(18, int(component.height * 0.55))
            expand_y = max(14, int(component.height * 0.48))
            attached = [
                fragment
                for fragment in small
                if original_x0 - expand_x <= fragment.cx <= original_x1 + expand_x
                and original_y0 - expand_y <= fragment.cy <= original_y1 + expand_y
            ]
            if attached:
                component.x0 = min([original_x0] + [fragment.x0 for fragment in attached])
                component.y0 = min([original_y0] + [fragment.y0 for fragment in attached])
                component.x1 = max([original_x1] + [fragment.x1 for fragment in attached])
                component.y1 = max([original_y1] + [fragment.y1 for fragment in attached])

    components.sort(key=lambda item: (int(round(item.cy / args.line_step)), item.cx))
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int]] = []
    for index, component in enumerate(components, start=1):
        rows.append(
            {
                "index": index,
                "line_index": int(round(component.cy / args.line_step)) + 1,
                "x0": component.x0,
                "y0": component.y0,
                "x1": component.x1,
                "y1": component.y1,
            }
        )
    with (args.output / "prompt_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "line_index", "x0", "y0", "x1", "y1"])
        writer.writeheader()
        writer.writerows(rows)

    if args.original_image is not None:
        overlay = Image.open(args.original_image).convert("RGB").resize((restored.shape[1], restored.shape[0]), Image.Resampling.LANCZOS)
    else:
        overlay = Image.open(args.input_dir / "restored_dark_ocr.png").convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for row in rows:
        box = (row["x0"], row["y0"], row["x1"], row["y1"])
        draw.rectangle(box, outline=(0, 210, 110), width=2)
        draw.text((row["x0"] + 1, row["y0"] + 1), f"{row['index']:03d}", fill=(0, 105, 55), font=font)
    overlay.save(args.output / "prompt_overlay.png")
    print(f"connected_components={count - 1} main_prompts={len(rows)} detached_fragments={len(small)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
