from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supraspinatus_locator.evaluation.metrics import evaluate_roi_prediction


class TestMetrics(unittest.TestCase):
    def test_roi_recall(self) -> None:
        pred = np.zeros((8, 8, 8), dtype=np.uint8)
        target = np.zeros_like(pred)
        pred[1:6, 1:6, 1:6] = 1
        target[2:4, 2:4, 2:4] = 1
        metrics = evaluate_roi_prediction(pred, target)
        self.assertEqual(metrics["roi_recall"], 1.0)
        self.assertGreater(metrics["search_space_reduction"], 1.0)


if __name__ == "__main__":
    unittest.main()
