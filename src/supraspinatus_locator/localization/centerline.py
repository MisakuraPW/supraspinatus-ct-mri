from __future__ import annotations

import numpy as np

from .roi_geometry import BBox3D


def centerline_heatmap_from_bbox(shape: tuple[int, int, int], bbox: BBox3D, sigma_voxels: float = 3.0) -> np.ndarray:
    """Create a simple straight centerline heatmap inside a candidate ROI bbox."""

    heatmap = np.zeros(shape, dtype=np.float32)
    x0, y0, z0 = bbox.min
    x1, y1, z1 = bbox.max
    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    cy = (y0 + y1) / 2.0
    cz = (z0 + z1) / 2.0
    yy = np.arange(shape[1], dtype=np.float32)
    zz = np.arange(shape[2], dtype=np.float32)
    y_grid, z_grid = np.meshgrid(yy, zz, indexing="ij")
    dist2 = (y_grid - cy) ** 2 + (z_grid - cz) ** 2
    tube = np.exp(-dist2 / max(2.0 * sigma_voxels * sigma_voxels, 1e-6))
    for x in xs.astype(int):
        heatmap[x, :, :] = np.maximum(heatmap[x, :, :], tube)
    return heatmap

