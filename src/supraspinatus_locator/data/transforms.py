from __future__ import annotations

import numpy as np


def normalize_minmax(data: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    x = data.astype(np.float32)
    lo = float(np.min(x) if lower is None else lower)
    hi = float(np.max(x) if upper is None else upper)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def ensure_3d(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    return data

