from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supraspinatus_locator.data.nifti_io import load_nifti, save_nifti_like


class TestNiftiIO(unittest.TestCase):
    def test_nifti_roundtrip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            path = Path(d) / "x.nii.gz"
            save_nifti_like(path, data, spacing=(0.5, 0.6, 1.0))
            loaded = load_nifti(path)
            self.assertEqual(loaded.data.shape, data.shape)
            self.assertTrue(np.allclose(loaded.spacing[:3], (0.5, 0.6, 1.0)))
            self.assertTrue(np.array_equal(loaded.data, data))


if __name__ == "__main__":
    unittest.main()
