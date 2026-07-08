from __future__ import annotations

import numpy as np


def spacing_scale(source_spacing: tuple[float, ...], target_spacing: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(s) / float(t) for s, t in zip(source_spacing, target_spacing))


def nearest_resample(volume: np.ndarray, source_spacing: tuple[float, ...], target_spacing: tuple[float, ...]) -> np.ndarray:
    """Dependency-light nearest-neighbor resampling for masks and quick prototypes."""

    scale = spacing_scale(source_spacing, target_spacing)
    new_shape = tuple(max(1, int(round(volume.shape[i] * scale[i]))) for i in range(volume.ndim))
    grids = [np.clip((np.arange(n) / scale[i]).round().astype(int), 0, volume.shape[i] - 1) for i, n in enumerate(new_shape)]
    return volume[np.ix_(*grids)]

