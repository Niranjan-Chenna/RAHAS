from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rahas_comparison_v1 import DATASET_SHA256, EXPERIMENTS, OUTPUT_ROOT, SPLIT_DIR, assert_frozen_dataset, sha256


EXPECTED_VALIDATION = 238
EXPECTED_TEST = 239
EXPECTED_EPOCHS = 12
PROPOSED_CHECKPOINT_SHA256 = "61290fe62bdad6c4b194f4a188800a376852b446614418ae3c7c62266cd270bd"


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_experiment(root: Path, experiment_id: str) -> dict:
    directory = root / experiment_id
    metrics_path = directory / "metrics.json"
    preflight_path = directory / "preflight.json"
    checkpoint_path = directory / "best.pt"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    epoch_count = len(csv_rows(directory / "epoch_metrics.csv"))
    validation_count = len(csv_rows(directory / "predictions_validation.csv"))
    test_count = len(csv_rows(directory / "predictions_test.csv"))
    checks = {
        "status_pass": result["status"] == "PASS",
        "dataset_hash": result["dataset_sha256"] == DATASET_SHA256 == preflight["dataset_sha256"],
        "leakage_preflight_pass": preflight["leakage_status"] == "PASS",
        "seed_2026": result["seed"] == 2026,
        "epoch_count_12": epoch_count == EXPECTED_EPOCHS,
        "validation_predictions_238": validation_count == EXPECTED_VALIDATION,
        "test_predictions_239": test_count == EXPECTED_TEST,
        "checkpoint_hash": sha256(checkpoint_path) == result["checkpoint_sha256"],
        "command_recorded": (directory / "command.txt").is_file(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "best_epoch": result["best_epoch"],
        "checkpoint_sha256": result["checkpoint_sha256"],
    }


def verify_reference(root: Path) -> dict:
    directory = root / "P0_rahas_proposed_reference"
    result = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    index_rows = csv_rows(directory / "base_confusion_sheet_index_v2.csv")
    confusion_rows = csv_rows(directory / "base_confusion_pairs_top20_v2.csv")
    review_rows = csv_rows(directory / "base_confusion_qualitative_review_v2.csv")
    yi_rows = [row for row in index_rows if row["class_label"] == "यी"]
    checks = {
        "status_pass": result["status"] == "PASS",
        "dataset_hash": result["dataset_sha256"] == DATASET_SHA256,
        "checkpoint_hash": result["checkpoint_sha256"] == PROPOSED_CHECKPOINT_SHA256,
        "validation_predictions_238": len(csv_rows(directory / "predictions_validation.csv")) == EXPECTED_VALIDATION,
        "test_predictions_239": len(csv_rows(directory / "predictions_test.csv")) == EXPECTED_TEST,
        "twenty_confusion_pairs": len(confusion_rows) == 20,
        "twenty_qualitative_reviews": len(review_rows) == 20,
        "indexed_tiles_present": len(index_rows) > 0,
        "verified_yii_preserved": bool(yi_rows) and all(row["verified_devanagari_full_label"] == "true" for row in yi_rows),
        "plain_base_sheets_present": (directory / "base_confusion_sheets_plain_base_v2").is_dir(),
        "overview_present": (directory / "base_confusion_overview_v2.png").is_file(),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> None:
    root = OUTPUT_ROOT.resolve()
    split = assert_frozen_dataset(SPLIT_DIR.resolve())
    experiments = {experiment_id: verify_experiment(root, experiment_id) for experiment_id in EXPERIMENTS}
    reference = verify_reference(root)
    table_rows = csv_rows(root / "comparison_table.csv")
    overall_checks = {
        "frozen_split_pass": split["leakage_status"] == "PASS",
        "all_trainable_experiments_pass": all(item["status"] == "PASS" for item in experiments.values()),
        "reference_pass": reference["status"] == "PASS",
        "comparison_table_13_rows": len(table_rows) == 13,
        "raw_restored_partial_recorded": any(
            row["experiment_id"] == "A6_raw_vs_restored" and row["status"] == "PARTIAL" for row in table_rows
        ),
    }
    payload = {
        "status": "PASS" if all(overall_checks.values()) else "FAIL",
        "dataset_sha256": DATASET_SHA256,
        "overall_checks": overall_checks,
        "reference": reference,
        "experiments": experiments,
        "superseded_artifacts": {
            "base_confusion_sheets": "Superseded by indexed v2 sheets because v1 mixed full modifier-bearing forms without an explicit index.",
            "base_confusion_pairs_top20.csv": "Superseded by base_confusion_pairs_top20_v2.csv.",
        },
    }
    (root / "final_verification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "overall_checks": overall_checks}, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
