from __future__ import annotations

import numpy as np

from supraspinatus_locator.localization.roi_geometry import bbox_from_mask, bbox_iou


def evaluate_roi_prediction(pred_mask: np.ndarray, target_mask: np.ndarray | None) -> dict[str, float]:
    pred = np.asarray(pred_mask) > 0
    metrics: dict[str, float] = {
        "pred_voxels": float(np.count_nonzero(pred)),
        "image_voxels": float(pred.size),
        "pred_volume_fraction": float(np.count_nonzero(pred) / pred.size) if pred.size else 0.0,
        "search_space_reduction": float(pred.size / max(np.count_nonzero(pred), 1)),
    }
    if target_mask is None:
        return metrics
    target = np.asarray(target_mask) > 0
    overlap = np.count_nonzero(pred & target)
    target_voxels = np.count_nonzero(target)
    pred_voxels = np.count_nonzero(pred)
    metrics.update(
        {
            "target_voxels": float(target_voxels),
            "overlap_voxels": float(overlap),
            "roi_recall": float(overlap / target_voxels) if target_voxels else 0.0,
            "roi_precision": float(overlap / pred_voxels) if pred_voxels else 0.0,
            "bbox_iou": bbox_iou(bbox_from_mask(pred), bbox_from_mask(target)),
        }
    )
    return metrics

