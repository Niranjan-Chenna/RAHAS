from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.source_disjoint_split import (
    apply_splits,
    difference_hash,
    extract_original_identity,
    file_sha256,
    leakage_audit,
    parse_reference,
)


class CanonicalIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.words = {
            "WORD_0001": {
                "reference": "PEDT, I, L. 4",
                "inscription_id": "PEDT",
                "page_id": "PEDT_I",
            }
        }

    def test_reference_parser(self) -> None:
        self.assertEqual(parse_reference("PEDT, I, L. 4"), ("PEDT", "PEDT_I"))
        self.assertEqual(parse_reference("PELA, V L. 5"), ("PELA", "PELA_V"))
        self.assertEqual(parse_reference("Unknown"), ("", ""))

    def test_word_identity(self) -> None:
        row = extract_original_identity("अ", "word_0001_char_01_0905.png", self.words)
        self.assertEqual(row["inscription_id"], "PEDT")
        self.assertEqual(row["page_id"], "PEDT_I")
        self.assertEqual(row["word_id"], "WORD_0001")
        self.assertEqual(row["original_crop_id"], "WORD_0001_CHAR_01")

    def test_page_and_crop_filename_families(self) -> None:
        col = extract_original_identity("क", "col_1_p001.png", self.words)
        self.assertEqual(col["page_id"], "REFERENCE_CHART_P001")
        diff_page = extract_original_identity("की", "diff_page_39_p001.png", self.words)
        self.assertEqual(diff_page["page_id"], "DIFF_PAGE_0039")
        diff_label = extract_original_identity("अ", "diff_अ_p001.png", self.words)
        self.assertEqual(diff_label["page_id"], "")
        self.assertEqual(diff_label["original_crop_id"], "DIFF_U0905_P001")


class SplitIntegrationTests(unittest.TestCase):
    def make_image(self, path: Path, seed: int) -> None:
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        Image.fromarray(data, mode="L").save(path)

    def make_original(self, path: Path, label: str, index: int) -> dict[str, str]:
        crop_id = f"CROP_{index:03d}"
        return {
            "sample_path": path.as_posix(),
            "class_label": label,
            "class_index": "1" if label == "A" else "2",
            "estampage_id": "",
            "inscription_id": "",
            "page_id": "",
            "word_id": f"WORD_{index:04d}",
            "original_crop_id": crop_id,
            "augmentation_parent_id": crop_id,
            "augmented_sample_id": "",
            "is_augmented": "false",
            "augmentation_type": "none",
            "source_family": "test",
            "source_reference": "",
            "identity_confidence": "test",
            "grouping_level": "word",
            "source_group_id": f"WORD::{index:04d}",
            "split_group_id": "",
            "file_sha256": file_sha256(path),
            "perceptual_hash": difference_hash(path),
            "split": "",
        }

    def test_constructed_split_has_zero_leakage_and_train_only_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(12):
                path = root / f"original_{index}.png"
                self.make_image(path, index + 100)
                rows.append(self.make_original(path, "A" if index % 2 == 0 else "B", index))
            originals = list(rows)
            for index, parent in enumerate(originals):
                path = root / f"augmented_{index}.png"
                self.make_image(path, index + 1000)
                augmented = dict(parent)
                augmented.update(
                    sample_path=path.as_posix(),
                    augmented_sample_id=f"AUG_{index:03d}",
                    is_augmented="true",
                    augmentation_type="test_augmentation",
                    file_sha256=file_sha256(path),
                    perceptual_hash=difference_hash(path),
                )
                rows.append(augmented)
            apply_splits(rows, seed=2026, perceptual_threshold=2)
            audit = leakage_audit(rows, perceptual_threshold=2)
            self.assertEqual(audit["status"], "PASS")
            self.assertTrue(any(row["split"] == "validation" for row in rows))
            self.assertTrue(any(row["split"] == "test" for row in rows))
            self.assertTrue(all(
                row["is_augmented"] == "false"
                for row in rows
                if row["split"] in {"validation", "test"}
            ))
            self.assertTrue(all(
                row["split"] == "train"
                for row in rows
                if row["is_augmented"] == "true" and row["split"] in {"train", "validation", "test"}
            ))

    def test_audit_fails_on_word_crossing(self) -> None:
        rows = [
            {
                "sample_path": "a.png", "split": "train", "file_sha256": "a", "perceptual_hash": "0000000000000000",
                "augmentation_parent_id": "A", "original_crop_id": "A", "word_id": "WORD_1", "page_id": "",
                "inscription_id": "", "estampage_id": "", "split_group_id": "A",
            },
            {
                "sample_path": "b.png", "split": "test", "file_sha256": "b", "perceptual_hash": "ffffffffffffffff",
                "augmentation_parent_id": "B", "original_crop_id": "B", "word_id": "WORD_1", "page_id": "",
                "inscription_id": "", "estampage_id": "", "split_group_id": "B",
            },
        ]
        audit = leakage_audit(rows, perceptual_threshold=2)
        self.assertEqual(audit["status"], "FAIL")
        self.assertFalse(audit["checks"]["word_ids"])


if __name__ == "__main__":
    unittest.main()

