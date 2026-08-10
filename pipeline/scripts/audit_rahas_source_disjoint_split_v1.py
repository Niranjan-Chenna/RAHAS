from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.source_disjoint_split import (
    ACTIVE_SPLITS,
    difference_hash,
    file_sha256,
    leakage_audit,
    stable_dataset_hash,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-audit an immutable RAHAS source-disjoint split.")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "datasets/splits/rahas_source_disjoint_v1",
    )
    parser.add_argument("--verify-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.experiment / "canonical_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((args.experiment / "split_summary.json").read_text(encoding="utf-8"))
    threshold = int(summary["perceptual_hamming_distance_threshold"])
    audit = leakage_audit(rows, threshold)
    extra_checks = {
        "validation_original_only": all(
            row["is_augmented"] == "false" for row in rows if row["split"] == "validation"
        ),
        "test_original_only": all(
            row["is_augmented"] == "false" for row in rows if row["split"] == "test"
        ),
        "active_splits_known": all(
            row["split"] in ACTIVE_SPLITS or row["split"].startswith("excluded_") for row in rows
        ),
        "dataset_hash_matches": stable_dataset_hash(rows) == summary["dataset_sha256"],
    }
    if args.verify_files:
        file_checks = []
        for row in rows:
            path = ROOT / row["sample_path"]
            file_checks.append(
                path.exists()
                and file_sha256(path) == row["file_sha256"]
                and difference_hash(path) == row["perceptual_hash"]
            )
        extra_checks["sample_files_match_hashes"] = all(file_checks)
    audit["checks"].update(extra_checks)
    audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    output = args.experiment / "leakage_audit.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "checks": audit["checks"]}, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

