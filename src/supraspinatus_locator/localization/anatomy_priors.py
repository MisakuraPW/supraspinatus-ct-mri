from __future__ import annotations

import numpy as np

from .roi_geometry import bbox_from_mask


def infer_roi_center_from_bone_centroid(
    bone_mask: np.ndarray,
    spacing: tuple[float, float, float],
    offset_mm: tuple[float, float, float] = (0.0, -6.0, 0.0),
) -> np.ndarray:
    """Estimate a shoulder tendon ROI center from the retained bone mask.

    This intentionally conservative heuristic is a baseline. It uses the bone
    cloud center and shifts slightly toward the expected subacromial soft tissue
    corridor in voxel coordinates. Later work should replace it with explicit
    scapula/humerus landmarks.
    """

    coords = np.argwhere(bone_mask)
    if coords.size == 0:
        raise ValueError("Cannot infer ROI center from an empty bone mask")
    center = coords.mean(axis=0)
    offset_vox = np.asarray(offset_mm, dtype=np.float32) / np.asarray(spacing[:3], dtype=np.float32)
    return center + offset_vox


def infer_roi_center_from_bone_bbox_fraction(
    bone_mask: np.ndarray,
    fraction_xyz: tuple[float, float, float] = (0.84, 0.10, 0.52),
) -> np.ndarray:
    """Estimate ROI center from a fraction inside the retained shoulder bone bbox.

    For the current LHY shoulder CT orientation, the supraspinatus ROI sits near
    the lateral-superior soft-tissue corridor relative to the retained bone
    envelope. This is deliberately exposed as a configurable baseline prior.
    """

    bbox = bbox_from_mask(bone_mask)
    if bbox is None:
        raise ValueError("Cannot infer ROI center from an empty bone mask")
    mn = np.asarray(bbox.min, dtype=np.float32)
    mx = np.asarray(bbox.max, dtype=np.float32)
    frac = np.asarray(fraction_xyz, dtype=np.float32)
    return mn + frac * (mx - mn)


def roi_size_mm_to_voxels(size_mm: tuple[float, float, float], spacing: tuple[float, float, float]) -> np.ndarray:
    return np.maximum(1, np.asarray(size_mm, dtype=np.float32) / np.asarray(spacing[:3], dtype=np.float32))
