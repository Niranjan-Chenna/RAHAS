from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.soft_data import build_image_records_from_manifest


class ManifestDataLoadingTests(unittest.TestCase):
    def test_manifest_loader_preserves_canonical_class_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            Image.new("L", (12, 12), 255).save(image_path)
            manifest_path = root / "train_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_path", "class_label", "class_index", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_path": "sample.png",
                        "class_label": "A",
                        "class_index": "1",
                        "split": "train",
                    }
                )
            maps = {
                "records": [
                    {
                        "label": "A",
                        "full_idx": 1,
                        "base_idx": 2,
                        "modifier_idx": 3,
                        "nasal": False,
                    }
                ]
            }
            records = build_image_records_from_manifest(manifest_path, maps, root, "train")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].path, image_path.resolve())
            self.assertEqual(records[0].full_idx, 1)

    def test_manifest_loader_rejects_wrong_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("L", (12, 12), 255).save(root / "sample.png")
            manifest_path = root / "validation_manifest.csv"
            manifest_path.write_text(
                "sample_path,class_label,class_index,split\n"
                "sample.png,A,1,train\n",
                encoding="utf-8",
            )
            maps = {
                "records": [
                    {
                        "label": "A",
                        "full_idx": 1,
                        "base_idx": 1,
                        "modifier_idx": 1,
                        "nasal": False,
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "expected 'validation'"):
                build_image_records_from_manifest(manifest_path, maps, root, "validation")


if __name__ == "__main__":
    unittest.main()
