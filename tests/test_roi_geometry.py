from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supraspinatus_locator.localization.roi_geometry import bbox_from_mask, bbox_iou, mask_from_bbox


class TestROIGeometry(unittest.TestCase):
    def test_bbox_from_mask_and_mask_from_bbox(self) -> None:
        mask = np.zeros((10, 10, 5), dtype=np.uint8)
        mask[2:5, 3:7, 1:3] = 1
        bbox = bbox_from_mask(mask)
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertEqual(bbox.min, (2, 3, 1))
        self.assertEqual(bbox.max, (4, 6, 2))
        self.assertTrue(np.array_equal(mask_from_bbox(mask.shape, bbox), mask))

    def test_bbox_iou(self) -> None:
        mask_a = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_b = np.zeros_like(mask_a)
        mask_a[0:5, 0:5, 0:5] = 1
        mask_b[2:7, 2:7, 2:7] = 1
        self.assertGreater(bbox_iou(bbox_from_mask(mask_a), bbox_from_mask(mask_b)), 0.0)
        self.assertLess(bbox_iou(bbox_from_mask(mask_a), bbox_from_mask(mask_b)), 1.0)


if __name__ == "__main__":
    unittest.main()
