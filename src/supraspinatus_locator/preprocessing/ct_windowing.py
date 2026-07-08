from __future__ import annotations

import numpy as np


def apply_window(volume: np.ndarray, center: float = 80.0, width: float = 500.0) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(volume.astype(np.float32), lo, hi)
    return ((x - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)


def auto_window(volume: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(volume, [low_percentile, high_percentile])
    x = np.clip(volume.astype(np.float32), lo, hi)
    return ((x - lo) / max(float(hi - lo), 1e-6) * 255.0).astype(np.uint8)

