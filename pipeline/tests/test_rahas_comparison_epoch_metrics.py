from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_rahas_comparison_v1 import mark_best_epoch


class EpochMetricsSchemaTests(unittest.TestCase):
    def test_marks_only_the_final_selected_checkpoint_epoch(self) -> None:
        rows = [
            {"epoch": 1, "validation_macro_f1": 0.20, "best": True},
            {"epoch": 2, "validation_macro_f1": 0.35, "best": True},
            {"epoch": 3, "validation_macro_f1": 0.35, "best": False},
        ]

        mark_best_epoch(rows, selected_epoch=2)

        self.assertEqual([row["best"] for row in rows], [False, True, False])
        self.assertEqual(sum(row["best"] for row in rows), 1)

    def test_rejects_missing_or_duplicate_selected_epoch_rows(self) -> None:
        with self.assertRaises(ValueError):
            mark_best_epoch([{"epoch": 1, "best": False}], selected_epoch=2)

        with self.assertRaises(ValueError):
            mark_best_epoch(
                [{"epoch": 1, "best": False}, {"epoch": 1, "best": False}],
                selected_epoch=1,
            )


if __name__ == "__main__":
    unittest.main()