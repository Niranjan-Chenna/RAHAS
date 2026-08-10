from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ACTIVE_SPLITS = ("train", "validation", "test")
MANIFEST_FIELDS = [
    "sample_path",
    "class_label",
    "class_index",
    "estampage_id",
    "inscription_id",
    "page_id",
    "word_id",
    "original_crop_id",
    "augmentation_parent_id",
    "augmented_sample_id",
    "is_augmented",
    "augmentation_type",
    "source_family",
    "source_reference",
    "identity_confidence",
    "grouping_level",
    "source_group_id",
    "split_group_id",
    "file_sha256",
    "perceptual_hash",
    "split",
]

KNOWN_INSCRIPTIONS = (
    "MPDIAKQ",
    "MPESN",
    "MREBA",
    "MREBR",
    "MREB",
    "PEDT",
    "PELA",
    "PELN",
    "PERP",
    "REG",
    "RED",
    "REJ",
    "REK",
    "CDBB",
    "RE",
)


def normalized_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def label_code(label: str) -> str:
    return "_".join(f"U{ord(character):04X}" for character in label)


def parse_reference(reference: str) -> tuple[str, str]:
    normalized = normalized_id(reference)
    inscription = next(
        (name for name in KNOWN_INSCRIPTIONS if normalized == name or normalized.startswith(name + "_")),
        "",
    )
    if not inscription:
        return "", ""
    parts = [part.strip() for part in reference.split(",")]
    if len(parts) < 2:
        return inscription, ""
    page_token = re.split(r"\bL\s*\.?\s*\d+", parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
    page_token = normalized_id(page_token)
    return inscription, f"{inscription}_{page_token}" if page_token else ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path) -> str:
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(resized.get_flattened_data())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def perceptual_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def portable_sample_path(path: Path) -> str:
    parts = path.parts
    if "datasets" in parts:
        return Path(*parts[parts.index("datasets") :]).as_posix()
    return path.as_posix()


def load_word_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = {}
    for row in rows:
        word_id = f"WORD_{int(row['S.No']):04d}"
        inscription_id, page_id = parse_reference(row.get("Reference", ""))
        metadata[word_id] = {
            "reference": row.get("Reference", "").strip(),
            "inscription_id": inscription_id,
            "page_id": page_id,
        }
    return metadata


def load_augmentation_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        output_name = Path(row["output_path"]).name
        result[(row["label"], output_name)] = {
            "parent_name": Path(row["source_path"]).name,
            "seed": row.get("seed", ""),
            "synthetic_index": row.get("synthetic_index", ""),
        }
    return result


def extract_original_identity(
    class_label: str,
    filename: str,
    word_metadata: dict[str, dict[str, str]],
) -> dict[str, str]:
    stem = Path(filename).stem
    match = re.match(r"word_(\d+)_char_(\d+)_", stem, re.IGNORECASE)
    if match:
        word_id = f"WORD_{int(match.group(1)):04d}"
        crop_id = f"{word_id}_CHAR_{int(match.group(2)):02d}"
        metadata = word_metadata.get(word_id, {})
        return {
            "estampage_id": "",
            "inscription_id": metadata.get("inscription_id", ""),
            "page_id": metadata.get("page_id", ""),
            "word_id": word_id,
            "original_crop_id": crop_id,
            "source_family": "verified_word_crop",
            "source_reference": metadata.get("reference", ""),
            "identity_confidence": "verified_manifest",
        }

    match = re.match(r"col_(\d+)_p(\d+)$", stem, re.IGNORECASE)
    if match:
        return {
            "estampage_id": "",
            "inscription_id": "",
            "page_id": f"REFERENCE_CHART_P{int(match.group(2)):03d}",
            "word_id": "",
            "original_crop_id": f"COL_{int(match.group(1)):02d}_P{int(match.group(2)):03d}",
            "source_family": "reference_chart_crop",
            "source_reference": filename,
            "identity_confidence": "filename_page_crop",
        }

    match = re.match(r"diff_page_(\d+)_p(\d+)$", stem, re.IGNORECASE)
    if match:
        return {
            "estampage_id": "",
            "inscription_id": "",
            "page_id": f"DIFF_PAGE_{int(match.group(1)):04d}",
            "word_id": "",
            "original_crop_id": f"DIFF_PAGE_{int(match.group(1)):04d}_P{int(match.group(2)):03d}",
            "source_family": "difference_page_crop",
            "source_reference": filename,
            "identity_confidence": "filename_page_crop",
        }

    match = re.match(r"diff_.+_p(\d+)$", stem, re.IGNORECASE)
    if match:
        return {
            "estampage_id": "",
            "inscription_id": "",
            "page_id": "",
            "word_id": "",
            "original_crop_id": f"DIFF_{label_code(class_label)}_P{int(match.group(1)):03d}",
            "source_family": "difference_label_crop",
            "source_reference": filename,
            "identity_confidence": "filename_crop_only",
        }

    raise ValueError(f"Unsupported original filename family: {class_label}/{filename}")


def strongest_source_group(row: dict[str, str]) -> tuple[str, str]:
    for level, field in (
        ("estampage", "estampage_id"),
        ("inscription", "inscription_id"),
        ("page", "page_id"),
        ("word", "word_id"),
        ("crop", "original_crop_id"),
    ):
        if row.get(field):
            return level, f"{level.upper()}::{row[field]}"
    raise ValueError(f"No canonical source identity for {row.get('sample_path')}")


def build_canonical_rows(
    character_root: Path,
    verified_characters: Path,
    augmentation_manifest: Path,
) -> list[dict[str, str]]:
    labels = sorted(path.name for path in character_root.iterdir() if path.is_dir())
    class_indices = {label: index for index, label in enumerate(labels, start=1)}
    word_metadata = load_word_metadata(verified_characters)
    augmentation_map = load_augmentation_manifest(augmentation_manifest)
    image_paths = sorted(
        path
        for path in character_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    originals: dict[tuple[str, str], dict[str, str]] = {}
    rows: list[dict[str, str]] = []

    for path in image_paths:
        label = path.parent.name
        if path.name.startswith("existing_wordstyle_aug_v1_"):
            continue
        identity = extract_original_identity(label, path.name, word_metadata)
        level, group_id = strongest_source_group(identity)
        row = {
            "sample_path": portable_sample_path(path),
            "class_label": label,
            "class_index": str(class_indices[label]),
            **identity,
            "augmentation_parent_id": identity["original_crop_id"],
            "augmented_sample_id": "",
            "is_augmented": "false",
            "augmentation_type": "none",
            "grouping_level": level,
            "source_group_id": group_id,
            "split_group_id": "",
            "file_sha256": file_sha256(path),
            "perceptual_hash": difference_hash(path),
            "split": "",
        }
        originals[(label, path.name)] = row
        rows.append(row)

    for path in image_paths:
        if not path.name.startswith("existing_wordstyle_aug_v1_"):
            continue
        label = path.parent.name
        provenance = augmentation_map.get((label, path.name))
        if provenance is None:
            raise ValueError(f"No augmentation manifest row for {label}/{path.name}")
        parent = originals.get((label, provenance["parent_name"]))
        if parent is None:
            raise ValueError(f"No original parent {label}/{provenance['parent_name']} for {path.name}")
        augmented_id = (
            f"AUG_EXISTING_WORDSTYLE_{label_code(label)}_"
            f"{int(provenance['synthetic_index']):03d}_{provenance['seed']}"
        )
        row = {
            **{field: parent[field] for field in (
                "estampage_id", "inscription_id", "page_id", "word_id", "original_crop_id",
                "source_family", "source_reference", "identity_confidence", "grouping_level", "source_group_id"
            )},
            "sample_path": portable_sample_path(path),
            "class_label": label,
            "class_index": str(class_indices[label]),
            "augmentation_parent_id": parent["original_crop_id"],
            "augmented_sample_id": augmented_id,
            "is_augmented": "true",
            "augmentation_type": "existing_character_wordstyle_v1",
            "split_group_id": "",
            "file_sha256": file_sha256(path),
            "perceptual_hash": difference_hash(path),
            "split": "",
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["class_index"]), row["sample_path"]))


class UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def merge_duplicate_source_groups(originals: list[dict[str, str]], threshold: int) -> None:
    source_groups = {row["source_group_id"] for row in originals}
    union = UnionFind(source_groups)
    by_exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in originals:
        by_exact[row["file_sha256"]].append(row)
    for duplicate_rows in by_exact.values():
        for row in duplicate_rows[1:]:
            union.union(duplicate_rows[0]["source_group_id"], row["source_group_id"])

    for index, first in enumerate(originals):
        for second in originals[index + 1 :]:
            if first["source_group_id"] == second["source_group_id"]:
                continue
            if perceptual_distance(first["perceptual_hash"], second["perceptual_hash"]) <= threshold:
                union.union(first["source_group_id"], second["source_group_id"])

    components: dict[str, list[str]] = defaultdict(list)
    for source_group in source_groups:
        components[union.find(source_group)].append(source_group)
    component_ids = {}
    for members in components.values():
        payload = "\n".join(sorted(members)).encode("utf-8")
        identifier = "SOURCE_COMPONENT::" + hashlib.sha256(payload).hexdigest()[:16]
        for member in members:
            component_ids[member] = identifier
    for row in originals:
        row["split_group_id"] = component_ids[row["source_group_id"]]


def assign_original_splits(originals: list[dict[str, str]], seed: int) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in originals:
        grouped[row["split_group_id"]].append(row)
    assignment = {group: "train" for group in grouped}
    train_counts = Counter(row["class_label"] for row in originals)
    destination_classes = {"validation": set(), "test": set()}
    destination_counts = {"validation": 0, "test": 0}
    target = max(1, round(len(originals) * 0.15))
    rng = random.Random(seed)
    tie_order = list(grouped)
    rng.shuffle(tie_order)
    tie_rank = {group: index for index, group in enumerate(tie_order)}

    while True:
        destinations = sorted(
            destination_counts,
            key=lambda name: (destination_counts[name] / target, name),
        )
        destination = destinations[0]
        if all(destination_counts[name] >= target for name in destination_counts):
            break
        candidates = []
        for group, group_rows in grouped.items():
            if assignment[group] != "train":
                continue
            group_counts = Counter(row["class_label"] for row in group_rows)
            if any(train_counts[label] - count < 1 for label, count in group_counts.items()):
                continue
            classes = set(group_counts)
            new_classes = len(classes - destination_classes[destination])
            projected = destination_counts[destination] + len(group_rows)
            overshoot = max(0, projected - target)
            candidates.append((new_classes, -overshoot, len(group_rows), -tie_rank[group], group))
        if not candidates:
            break
        group = max(candidates)[-1]
        assignment[group] = destination
        moved = grouped[group]
        moved_counts = Counter(row["class_label"] for row in moved)
        train_counts.subtract(moved_counts)
        destination_counts[destination] += len(moved)
        destination_classes[destination].update(moved_counts)
    return assignment


def apply_splits(rows: list[dict[str, str]], seed: int, perceptual_threshold: int) -> None:
    originals = [row for row in rows if row["is_augmented"] == "false"]
    merge_duplicate_source_groups(originals, perceptual_threshold)
    assignments = assign_original_splits(originals, seed)
    parent_splits = {}
    for row in originals:
        row["split"] = assignments[row["split_group_id"]]
        parent_splits[row["original_crop_id"]] = row["split"]
    original_by_parent = {row["original_crop_id"]: row for row in originals}
    held_out = [row for row in originals if row["split"] in {"validation", "test"}]
    for row in rows:
        if row["is_augmented"] == "false":
            continue
        parent = original_by_parent[row["augmentation_parent_id"]]
        row["split_group_id"] = parent["split_group_id"]
        if parent_splits[row["augmentation_parent_id"]] != "train":
            row["split"] = "excluded_parent_not_train"
            continue
        collision = any(
            row["file_sha256"] == other["file_sha256"]
            or perceptual_distance(row["perceptual_hash"], other["perceptual_hash"]) <= perceptual_threshold
            for other in held_out
        )
        row["split"] = "excluded_perceptual_collision" if collision else "train"


def crossing_values(rows: list[dict[str, str]], field: str) -> dict[str, list[str]]:
    splits_by_value: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(field, "")
        if value and row["split"] in ACTIVE_SPLITS:
            splits_by_value[value].add(row["split"])
    return {value: sorted(splits) for value, splits in splits_by_value.items() if len(splits) > 1}


def leakage_audit(rows: list[dict[str, str]], perceptual_threshold: int) -> dict:
    active = [row for row in rows if row["split"] in ACTIVE_SPLITS]
    exact = crossing_values(active, "file_sha256")
    perceptual_pairs = []
    for index, first in enumerate(active):
        for second in active[index + 1 :]:
            if first["split"] == second["split"]:
                continue
            distance = perceptual_distance(first["perceptual_hash"], second["perceptual_hash"])
            if distance <= perceptual_threshold:
                perceptual_pairs.append({
                    "first": first["sample_path"],
                    "first_split": first["split"],
                    "second": second["sample_path"],
                    "second_split": second["split"],
                    "distance": distance,
                })
    rules = {
        "file_hashes": exact,
        "perceptual_hashes": perceptual_pairs,
        "augmentation_parents": crossing_values(active, "augmentation_parent_id"),
        "original_crop_ids": crossing_values(active, "original_crop_id"),
        "word_ids": crossing_values(active, "word_id"),
        "page_ids": crossing_values(active, "page_id"),
        "inscription_ids": crossing_values(active, "inscription_id"),
        "estampage_ids": crossing_values(active, "estampage_id"),
        "split_group_ids": crossing_values(active, "split_group_id"),
    }
    checks = {name: len(values) == 0 for name, values in rules.items()}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "perceptual_hash_algorithm": "64-bit difference hash",
        "perceptual_hamming_distance_threshold": perceptual_threshold,
        "checks": checks,
        "cross_split_counts": {name: len(values) for name, values in rules.items()},
        "violations": rules,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] = MANIFEST_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_dataset_hash(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(
        f"{row['split']}\t{row['sample_path']}\t{row['file_sha256']}"
        for row in sorted(rows, key=lambda item: (item["split"], item["sample_path"]))
        if row["split"] in ACTIVE_SPLITS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_summary(rows: list[dict[str, str]], audit: dict, seed: int, threshold: int) -> tuple[dict, list[dict[str, str]]]:
    labels = sorted({row["class_label"] for row in rows})
    originals = [row for row in rows if row["is_augmented"] == "false"]
    before_counts = Counter(row["split"] for row in originals)
    after_counts = Counter(row["split"] for row in rows)
    class_rows = []
    for label in labels:
        label_rows = [row for row in rows if row["class_label"] == label]
        counts = Counter(row["split"] for row in label_rows)
        original_counts = Counter(row["split"] for row in label_rows if row["is_augmented"] == "false")
        class_rows.append({
            "class_label": label,
            "class_index": next(row["class_index"] for row in label_rows),
            "train_original": str(original_counts["train"]),
            "train_augmented": str(counts["train"] - original_counts["train"]),
            "train_total": str(counts["train"]),
            "validation": str(counts["validation"]),
            "test": str(counts["test"]),
            "excluded_augmentations": str(sum(count for split, count in counts.items() if split.startswith("excluded_"))),
        })
    split_groups = {
        split: len({row["split_group_id"] for row in rows if row["split"] == split})
        for split in ACTIVE_SPLITS
    }
    represented = {
        split: sorted({row["class_label"] for row in rows if row["split"] == split})
        for split in ACTIVE_SPLITS
    }
    train_original_counts = Counter(row["class_label"] for row in originals if row["split"] == "train")
    identity_level_counts = Counter(row["grouping_level"] for row in originals)
    held_out_inscriptions = {
        split: sorted({row["inscription_id"] for row in originals if row["split"] == split and row["inscription_id"]})
        for split in ACTIVE_SPLITS
    }
    summary = {
        "experiment": "rahas_source_disjoint_v1",
        "seed": seed,
        "split_hierarchy": ["estampage_id", "inscription_id", "page_id", "word_id", "original_crop_id"],
        "split_grouping_field": "split_group_id",
        "split_group_construction": "strongest available canonical source identity, unioned by exact hash and dHash distance threshold",
        "identity_level_original_samples": dict(sorted(identity_level_counts.items())),
        "held_out_inscriptions": held_out_inscriptions,
        "hierarchy_decision": {
            "estampage_level": "impossible: 0 of 2875 originals have a defensible link to the 16 full estampage scans",
            "inscription_level": "used for every verified word crop with a parseable inscription reference",
            "page_level": "used when inscription identity is unavailable but page identity is encoded",
            "word_level": "used only when verified word identity exists but inscription/page is unresolved",
            "crop_level": "last-resort grouping for diff_<label>_p* originals lacking stronger metadata",
        },
        "augmentation_policy": "Originals were split first. Existing deterministic derivatives were admitted only for training parents; held-out-parent derivatives were excluded.",
        "perceptual_hamming_distance_threshold": threshold,
        "total_existing_samples": len(rows),
        "original_samples": len(originals),
        "augmented_samples": len(rows) - len(originals),
        "before_augmentation": {split: before_counts[split] for split in ACTIVE_SPLITS},
        "after_augmentation": {split: after_counts[split] for split in ACTIVE_SPLITS},
        "excluded_augmentations": {split: count for split, count in after_counts.items() if split.startswith("excluded_")},
        "source_groups_per_split": split_groups,
        "classes_per_split": {split: len(values) for split, values in represented.items()},
        "classes_absent_from_validation": sorted(set(labels) - set(represented["validation"])),
        "classes_absent_from_test": sorted(set(labels) - set(represented["test"])),
        "meaningful_372_class_test_claim": False,
        "episodic_evaluation_redesign_required": True,
        "one_shot_train_original_classes": sum(count == 1 for count in train_original_counts.values()),
        "low_shot_train_original_classes_le_4": sum(count <= 4 for count in train_original_counts.values()),
        "dataset_sha256": stable_dataset_hash(rows),
        "leakage_status": audit["status"],
        "unresolved_identity": {
            "estampage_mapping": "No reliable link from OCR crop corpus to datasets/original_estampages was found.",
            "difference_label_crops": "diff_<label>_p* provides crop identity only; no page or inscription field exists.",
        },
    }
    return summary, class_rows


def manifest_sha256(path: Path) -> str:
    return file_sha256(path)


def write_experiment(output: Path, rows: list[dict[str, str]], audit: dict, summary: dict, class_rows: list[dict[str, str]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    canonical = output / "canonical_manifest.csv"
    write_csv(canonical, rows)
    for split, filename in (("train", "train_manifest.csv"), ("validation", "validation_manifest.csv"), ("test", "test_manifest.csv")):
        write_csv(output / filename, [row for row in rows if row["split"] == split])
    write_csv(
        output / "class_distribution.csv",
        class_rows,
        ["class_label", "class_index", "train_original", "train_augmented", "train_total", "validation", "test", "excluded_augmentations"],
    )
    hashes = {
        name: manifest_sha256(output / name)
        for name in ("canonical_manifest.csv", "train_manifest.csv", "validation_manifest.csv", "test_manifest.csv", "class_distribution.csv")
    }
    summary["manifest_sha256"] = hashes
    (output / "leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
