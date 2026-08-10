from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.soft_data import build_image_records, build_training_label_maps, grouped_split
from src.ocr.source_disjoint_split import (
    ACTIVE_SPLITS,
    MANIFEST_FIELDS,
    apply_splits,
    build_canonical_rows,
    build_summary,
    file_sha256,
    leakage_audit,
    write_csv,
    write_experiment,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build immutable source-disjoint RAHAS OCR manifests.")
    parser.add_argument("--characters", type=Path, default=ROOT / "datasets/prepared/12_ocr_soft_resized_v1/characters")
    parser.add_argument("--verified-characters", type=Path, default=ROOT / "datasets/reference_files/verified_characters.csv")
    parser.add_argument(
        "--augmentation-manifest",
        type=Path,
        default=ROOT / "datasets/prepared/10_existing_character_wordstyle_aug_v1/existing_wordstyle_augmentation_manifest.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/splits/rahas_source_disjoint_v1")
    parser.add_argument("--legacy-output", type=Path, default=ROOT / "datasets/splits/legacy_leaky_split_experiment")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perceptual-distance", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prepare_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Output already exists and is not empty: {path}. Use --force to regenerate.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_legacy_experiment(
    output: Path,
    canonical_rows: list[dict[str, str]],
    character_root: Path,
    seed: int,
    force: bool,
) -> None:
    prepare_output(output, force)
    maps = build_training_label_maps(character_root)
    records = build_image_records(character_root, maps)
    train, validation, test = grouped_split(records, seed)
    split_by_path = {
        str(record.path.resolve()).lower(): split
        for split, split_records in (("train", train), ("validation", validation), ("test", test))
        for record in split_records
    }
    legacy_rows = []
    for source in canonical_rows:
        row = dict(source)
        row["split"] = split_by_path[str((ROOT / row["sample_path"]).resolve()).lower()]
        legacy_rows.append(row)
    write_csv(output / "legacy_split_manifest.csv", legacy_rows, MANIFEST_FIELDS)
    checkpoint = ROOT / "pipeline/checkpoints/rahas_spatial_proto_v1/best.pt"
    summary = {
        "experiment": "legacy_leaky_split_experiment",
        "status": "PRESERVED_INVALID_BASELINE",
        "warning": "Do not use for publication. This is the pre-correction grouped_split reconstruction.",
        "seed": seed,
        "samples_per_split": {
            split: sum(row["split"] == split for row in legacy_rows) for split in ACTIVE_SPLITS
        },
        "manifest_sha256": file_sha256(output / "legacy_split_manifest.csv"),
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix() if checkpoint.exists() else "",
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.exists() else "",
        "known_leakage": {
            "exact_file_hashes_crossing_splits": 1,
            "augmentation_origin_groups_crossing_splits": 120,
            "records_in_crossing_augmentation_origin_groups": 327,
            "word_ids_crossing_splits": 434,
        },
    }
    (output / "legacy_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.perceptual_distance <= 8:
        raise ValueError("--perceptual-distance must be between 0 and 8")
    prepare_output(args.output, args.force)
    rows = build_canonical_rows(args.characters, args.verified_characters, args.augmentation_manifest)
    write_legacy_experiment(args.legacy_output, rows, args.characters, args.seed, args.force)
    apply_splits(rows, args.seed, args.perceptual_distance)
    audit = leakage_audit(rows, args.perceptual_distance)
    summary, class_rows = build_summary(rows, audit, args.seed, args.perceptual_distance)
    write_experiment(args.output, rows, audit, summary, class_rows)
    print(
        f"status={audit['status']} originals={summary['original_samples']} "
        f"train={summary['after_augmentation']['train']} validation={summary['after_augmentation']['validation']} "
        f"test={summary['after_augmentation']['test']} dataset_sha256={summary['dataset_sha256']}",
        flush=True,
    )
    print(f"output={args.output.resolve()}", flush=True)
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

