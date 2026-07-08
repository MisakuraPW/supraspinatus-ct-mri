from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BBox3D:
    min: tuple[int, int, int]
    max: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(max(0, self.max[i] - self.min[i] + 1)) for i in range(3))

    @property
    def volume(self) -> int:
        s = self.shape
        return int(s[0] * s[1] * s[2])

    def to_dict(self) -> dict[str, list[int]]:
        return {"min": [int(v) for v in self.min], "max": [int(v) for v in self.max]}


def bbox_from_mask(mask: np.ndarray) -> BBox3D | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return BBox3D(
        tuple(int(v) for v in coords.min(axis=0)),
        tuple(int(v) for v in coords.max(axis=0)),
    )


def clamp_bbox(center: np.ndarray, size_voxels: np.ndarray, shape: tuple[int, int, int]) -> BBox3D:
    half = np.maximum(1, np.round(size_voxels / 2.0).astype(int))
    c = np.round(center).astype(int)
    mn = c - half
    mx = c + half
    for i in range(3):
        if mn[i] < 0:
            mx[i] -= mn[i]
            mn[i] = 0
        if mx[i] >= shape[i]:
            shift = mx[i] - shape[i] + 1
            mn[i] = max(0, mn[i] - shift)
            mx[i] = shape[i] - 1
    return BBox3D(tuple(mn.astype(int)), tuple(mx.astype(int)))


def mask_from_bbox(shape: tuple[int, int, int], bbox: BBox3D) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, z0 = bbox.min
    x1, y1, z1 = bbox.max
    mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = 1
    return mask


def bbox_iou(a: BBox3D | None, b: BBox3D | None) -> float:
    if a is None or b is None:
        return 0.0
    inter_min = [max(a.min[i], b.min[i]) for i in range(3)]
    inter_max = [min(a.max[i], b.max[i]) for i in range(3)]
    inter_shape = [max(0, inter_max[i] - inter_min[i] + 1) for i in range(3)]
    inter = inter_shape[0] * inter_shape[1] * inter_shape[2]
    union = a.volume + b.volume - inter
    return float(inter / union) if union else 0.0
