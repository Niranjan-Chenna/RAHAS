from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.ocr.label_decomposition import NONE, parse_label


UNKNOWN_LABEL = "_"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class OCRImageRecord:
    path: Path
    label: str
    full_idx: int
    base_idx: int
    modifier_idx: int
    nasal: float
    original_crop_id: str = ""
    augmentation_parent_id: str = ""
    word_id: str = ""
    inscription_id: str = ""
    estampage_id: str = ""
    is_augmented: bool = False


def _class_folders(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def build_training_label_maps(character_root: Path) -> dict:
    folders = _class_folders(character_root)
    if not folders:
        raise FileNotFoundError(f"No class folders found under {character_root}")
    parsed = [parse_label(folder.name) for folder in folders]
    full_labels = [UNKNOWN_LABEL, *sorted(record.full_label for record in parsed)]
    base_glyphs = [UNKNOWN_LABEL, *sorted({record.base_glyph for record in parsed})]
    modifiers = [UNKNOWN_LABEL, *sorted({record.modifier for record in parsed})]
    full_to_idx = {label: index for index, label in enumerate(full_labels)}
    base_to_idx = {label: index for index, label in enumerate(base_glyphs)}
    modifier_to_idx = {label: index for index, label in enumerate(modifiers)}
    records = [
        {
            "label": record.full_label,
            "base_glyph": record.base_glyph,
            "modifier": record.modifier,
            "nasal": record.nasal != NONE,
            "full_idx": full_to_idx[record.full_label],
            "base_idx": base_to_idx[record.base_glyph],
            "modifier_idx": modifier_to_idx[record.modifier],
        }
        for record in parsed
    ]
    return {
        "source": "src.ocr.label_decomposition.parse_label",
        "records": records,
        "full_label_to_idx": full_to_idx,
        "idx_to_full_label": full_labels,
        "base_glyph_to_idx": base_to_idx,
        "idx_to_base_glyph": base_glyphs,
        "modifier_to_idx": modifier_to_idx,
        "idx_to_modifier": modifiers,
        "unknown_label": UNKNOWN_LABEL,
    }


def build_image_records(character_root: Path, label_maps: dict) -> list[OCRImageRecord]:
    by_label = {record["label"]: record for record in label_maps["records"]}
    result = []
    for folder in _class_folders(character_root):
        info = by_label[folder.name]
        for path in sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            result.append(
                OCRImageRecord(
                    path=path,
                    label=folder.name,
                    full_idx=int(info["full_idx"]),
                    base_idx=int(info["base_idx"]),
                    modifier_idx=int(info["modifier_idx"]),
                    nasal=1.0 if info["nasal"] else 0.0,
                )
            )
    if not result:
        raise FileNotFoundError(f"No character images found under {character_root}")
    return result


def build_image_records_from_manifest(
    manifest_path: Path,
    label_maps: dict,
    project_root: Path,
    expected_split: str,
) -> list[OCRImageRecord]:
    by_label = {record["label"]: record for record in label_maps["records"]}
    manifest_path = manifest_path.resolve()
    project_root = project_root.resolve()
    result = []
    seen_paths = set()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            split = row.get("split", "")
            if split != expected_split:
                raise ValueError(
                    f"{manifest_path}:{row_number} has split={split!r}; expected {expected_split!r}"
                )
            label = row.get("class_label", "")
            if label not in by_label:
                raise ValueError(f"{manifest_path}:{row_number} has unknown class_label={label!r}")
            info = by_label[label]
            class_index = int(row["class_index"])
            if class_index != int(info["full_idx"]):
                raise ValueError(
                    f"{manifest_path}:{row_number} class_index={class_index} does not match "
                    f"the canonical label-map index {info['full_idx']} for {label!r}"
                )
            raw_path = Path(row["sample_path"])
            path = raw_path if raw_path.is_absolute() else project_root / raw_path
            path = path.resolve()
            if path in seen_paths:
                raise ValueError(f"Duplicate sample_path in {manifest_path}: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Manifest sample does not exist: {path}")
            seen_paths.add(path)
            result.append(
                OCRImageRecord(
                    path=path,
                    label=label,
                    full_idx=class_index,
                    base_idx=int(info["base_idx"]),
                    modifier_idx=int(info["modifier_idx"]),
                    nasal=1.0 if info["nasal"] else 0.0,
                    original_crop_id=row.get("original_crop_id", ""),
                    augmentation_parent_id=row.get("augmentation_parent_id", ""),
                    word_id=row.get("word_id", ""),
                    inscription_id=row.get("inscription_id", ""),
                    estampage_id=row.get("estampage_id", ""),
                    is_augmented=row.get("is_augmented", "").strip().lower() == "true",
                )
            )
    if not result:
        raise ValueError(f"No samples found in {manifest_path}")
    return result


def pil_to_gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def relative_darkness_array(gray: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    background = image.filter(ImageFilter.MaxFilter(21)).filter(ImageFilter.GaussianBlur(3.0))
    background_array = np.asarray(background, dtype=np.float32) / 255.0
    darkness = (background_array - gray) / np.maximum(background_array, 0.08)
    return np.clip(np.power(np.clip(darkness, 0.0, 1.0), 0.85), 0.0, 1.0)


def soft_bounds(gray: np.ndarray) -> tuple[int, int, int, int]:
    darkness = relative_darkness_array(gray)
    threshold = max(0.035, float(np.percentile(darkness, 95)) * 0.20)
    rows = np.where(darkness.max(axis=1) > threshold)[0]
    columns = np.where(darkness.max(axis=0) > threshold)[0]
    if rows.size == 0 or columns.size == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def fit_crop_onto_canvas(image: Image.Image, image_size: int, rng: random.Random | None, train: bool) -> Image.Image:
    x0, y0, x1, y1 = soft_bounds(pil_to_gray_array(image))
    pad = max(4, int(max(x1 - x0, y1 - y0) * 0.10))
    crop = image.convert("L").crop(
        (max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad))
    )
    fill = rng.uniform(0.70, 0.88) if train and rng is not None else 0.78
    scale = min((image_size * fill) / max(crop.width, 1), (image_size * fill) / max(crop.height, 1))
    crop = crop.resize((max(4, int(crop.width * scale)), max(4, int(crop.height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (image_size, image_size), 255)
    max_jitter = max(1, int(image_size * 0.04))
    jitter_x = rng.randint(-max_jitter, max_jitter) if train and rng is not None else 0
    jitter_y = rng.randint(-max_jitter, max_jitter) if train and rng is not None else 0
    canvas.paste(crop, ((image_size - crop.width) // 2 + jitter_x, (image_size - crop.height) // 2 + jitter_y))
    return canvas


def augment_grayscale(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("L").rotate(rng.uniform(-8.0, 8.0), expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
    if rng.random() < 0.50:
        factor = rng.uniform(0.82, 1.22)
        image = image.resize((max(8, int(image.width * factor)), max(8, int(image.height * factor))), Image.Resampling.BICUBIC)
    if rng.random() < 0.60:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.45))
    if rng.random() < 0.35:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.08))
    if rng.random() < 0.25:
        image = Image.blend(image, image.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.85))), rng.uniform(0.35, 0.85))
    if rng.random() < 0.25:
        filtered = image.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))
        image = Image.blend(image, filtered, rng.uniform(0.12, 0.32))
    return image


def soft_features(image: Image.Image, size: int, rng: random.Random | None = None, train: bool = False) -> torch.Tensor:
    if train and rng is not None:
        image = augment_grayscale(image, rng)
    gray = pil_to_gray_array(fit_crop_onto_canvas(image, size, rng, train))
    darkness = relative_darkness_array(gray)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    contrast = np.abs(gray - local_mean)
    contrast /= max(float(np.percentile(contrast, 99)), 1e-4)
    soft_alpha = np.power(np.clip(darkness, 0, 1), 0.72)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    gradient /= max(float(np.percentile(gradient, 99)), 1e-4)
    channels = np.stack([gray, darkness, np.clip(contrast, 0, 1), soft_alpha, np.clip(gradient, 0, 1)])
    return torch.from_numpy(channels.astype(np.float32))


def continuous_stroke_attenuation(features: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Create a softly faded view without thresholding or deleting pixels."""

    if features.ndim != 3 or features.shape[0] != 5:
        raise ValueError(f"expected 5xHxW features, got {tuple(features.shape)}")
    height, width = features.shape[-2:]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = rng.uniform(width * 0.20, width * 0.80)
    center_y = rng.uniform(height * 0.20, height * 0.80)
    sigma_x = rng.uniform(width * 0.12, width * 0.30)
    sigma_y = rng.uniform(height * 0.12, height * 0.30)
    strength = rng.uniform(0.25, 0.70)
    gaussian = np.exp(
        -0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2)
    )
    preservation = torch.from_numpy((1.0 - strength * gaussian).astype(np.float32))
    result = features.clone()
    result[0] = 1.0 - (1.0 - result[0]) * preservation
    result[1:] = result[1:] * preservation.unsqueeze(0)
    return result.clamp(0.0, 1.0)


class SoftFeatureDataset(Dataset):
    def __init__(self, records, size: int, train: bool, seed: int) -> None:
        self.records = records
        self.size = size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        rng = random.Random(self.seed + index * 1009) if self.train else None
        with Image.open(record.path) as image:
            features = soft_features(image.convert("L"), self.size, rng, self.train)
        return {
            "image": features,
            "full": torch.tensor(record.full_idx),
            "base": torch.tensor(record.base_idx),
            "modifier": torch.tensor(record.modifier_idx),
            "nasal": torch.tensor(record.nasal, dtype=torch.float32),
        }


def source_key(record: OCRImageRecord) -> str:
    stem = record.path.stem.lower()
    match = re.search(r"(col_\d+_p\d+|diff_[^_]*_p\d+|p\d+)", stem)
    key = match.group(1) if match else re.sub(r"(_seed_?\d+|_aug_?\d+|_\d{3,})+$", "", stem)
    return record.label + "::" + key


def grouped_split(records, seed: int):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.label][source_key(record)].append(record)
    train, validation, test = [], [], []
    rng = random.Random(seed)
    for groups in grouped.values():
        keys = list(groups)
        rng.shuffle(keys)
        count = len(keys)
        if count < 3:
            train_count = max(1, count - 1)
            validation_count = count - train_count
        else:
            train_count = min(max(1, int(round(count * 0.70))), count - 2)
            validation_count = max(1, int(round(count * 0.15)))
        train_keys = keys[:train_count]
        validation_keys = keys[train_count : train_count + validation_count]
        test_keys = keys[train_count + validation_count :]
        if not test_keys and validation_keys:
            test_keys = [validation_keys.pop()]
        for key in train_keys:
            train += groups[key]
        for key in validation_keys:
            validation += groups[key]
        for key in test_keys:
            test += groups[key]
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def macro_f1(prediction: torch.Tensor, target: torch.Tensor, classes: int) -> float:
    scores = []
    for class_index in range(classes):
        true_positive = ((prediction == class_index) & (target == class_index)).sum()
        false_positive = ((prediction == class_index) & (target != class_index)).sum()
        false_negative = ((prediction != class_index) & (target == class_index)).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator > 0:
            scores.append(float(2 * true_positive / denominator))
    return float(np.mean(scores)) if scores else 0.0
