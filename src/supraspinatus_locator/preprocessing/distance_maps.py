from __future__ import annotations

import numpy as np


def normalized_coordinate_channels(shape: tuple[int, int, int]) -> np.ndarray:
    """Return 3 coordinate channels normalized to [-1, 1]."""

    axes = [np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in shape]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=0)


def unsigned_distance_map(mask: np.ndarray, spacing: tuple[float, float, float] | None = None) -> np.ndarray:
    """Compute a physical unsigned distance map when scipy is available.

    If scipy is unavailable, this returns a zero map so the training pipeline can
    still be wired up and replaced on cloud machines with full dependencies.
    """

    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return np.zeros(mask.shape, dtype=np.float32)
    sampling = spacing if spacing is not None else (1.0, 1.0, 1.0)
    return distance_transform_edt(~(mask > 0), sampling=sampling).astype(np.float32)


def build_prior_channels(
    bone_mask: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
    include_coordinates: bool = True,
) -> np.ndarray:
    channels = [bone_mask.astype(np.float32)[None], unsigned_distance_map(bone_mask, spacing)[None]]
    if include_coordinates:
        channels.append(normalized_coordinate_channels(bone_mask.shape))
    return np.concatenate(channels, axis=0)

