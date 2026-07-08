from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supraspinatus_locator.evaluation.metrics import evaluate_roi_prediction
from supraspinatus_locator.preprocessing.body_bone_masks import largest_components, threshold_bone

from .anatomy_priors import (
    infer_roi_center_from_bone_bbox_fraction,
    infer_roi_center_from_bone_centroid,
    roi_size_mm_to_voxels,
)
from .roi_geometry import BBox3D, bbox_from_mask, clamp_bbox, mask_from_bbox


@dataclass
class RuleBasedConfig:
    bone_threshold_hu: float = 180.0
    min_bone_component_voxels: int = 5000
    max_bone_components: int = 3
    roi_size_mm: tuple[float, float, float] = (34.0, 18.0, 12.0)
    roi_offset_from_bone_center_mm: tuple[float, float, float] = (0.0, -6.0, 0.0)
    center_strategy: str = "bone_bbox_fraction"
    bone_bbox_fraction_xyz: tuple[float, float, float] = (0.84, 0.10, 0.52)


def locate_roi(volume: np.ndarray, spacing: tuple[float, float, float], config: RuleBasedConfig) -> tuple[np.ndarray, BBox3D, dict]:
    bone = threshold_bone(volume, config.bone_threshold_hu)
    retained_bone = largest_components(
        bone,
        min_voxels=config.min_bone_component_voxels,
        max_components=config.max_bone_components,
    )
    if not retained_bone.any():
        retained_bone = bone
    if config.center_strategy == "bone_centroid_offset":
        center = infer_roi_center_from_bone_centroid(retained_bone, spacing, config.roi_offset_from_bone_center_mm)
    elif config.center_strategy == "bone_bbox_fraction":
        center = infer_roi_center_from_bone_bbox_fraction(retained_bone, config.bone_bbox_fraction_xyz)
    else:
        raise ValueError(f"Unknown center_strategy: {config.center_strategy}")
    size_vox = roi_size_mm_to_voxels(config.roi_size_mm, spacing)
    bbox = clamp_bbox(center, size_vox, volume.shape)
    roi = mask_from_bbox(volume.shape, bbox)
    meta = {
        "method": "rule_based_bone_prior",
        "bone_threshold_hu": config.bone_threshold_hu,
        "min_bone_component_voxels": config.min_bone_component_voxels,
        "roi_size_mm": list(config.roi_size_mm),
        "roi_offset_from_bone_center_mm": list(config.roi_offset_from_bone_center_mm),
        "center_strategy": config.center_strategy,
        "bone_bbox_fraction_xyz": list(config.bone_bbox_fraction_xyz),
        "roi_center_voxel": [float(x) for x in center],
        "roi_bbox": bbox.to_dict(),
        "bone_bbox": bbox_from_mask(retained_bone).to_dict() if bbox_from_mask(retained_bone) else None,
    }
    return roi, bbox, meta


def save_localization_outputs(
    out_dir: str | Path,
    roi_mask: np.ndarray,
    bbox: BBox3D,
    meta: dict,
    target_mask: np.ndarray | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_roi_prediction(roi_mask, target_mask) if target_mask is not None else {}
    payload = {"bbox": bbox.to_dict(), "metadata": meta, "metrics": metrics}
    (out / "roi.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
