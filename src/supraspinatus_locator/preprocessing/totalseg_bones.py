from __future__ import annotations

from pathlib import Path

import numpy as np

from supraspinatus_locator.data.nifti_io import NiftiImage, load_nifti, save_nifti_like


SHOULDER_BONE_CLASSES = (
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
)


def load_mask_compatible(path: str | Path, image_shape: tuple[int, ...]) -> np.ndarray:
    mask = load_nifti(path).data > 0
    if mask.shape == image_shape:
        return mask
    if len(mask.shape) == 3 and len(image_shape) == 3 and mask.shape[:2] == (image_shape[1], image_shape[0]) and mask.shape[2] == image_shape[2]:
        return np.transpose(mask, (1, 0, 2))
    raise ValueError(f"Mask shape {mask.shape} is not compatible with image shape {image_shape}: {path}")


def load_totalseg_shoulder_bone_mask(seg_dir: str | Path, image_shape: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    seg_dir = Path(seg_dir)
    combined = np.zeros(image_shape, dtype=bool)
    loaded: list[str] = []
    for class_name in SHOULDER_BONE_CLASSES:
        path = seg_dir / f"{class_name}.nii.gz"
        if not path.exists():
            path = seg_dir / f"{class_name}.nii"
        if not path.exists():
            continue
        combined |= load_mask_compatible(path, image_shape)
        loaded.append(class_name)
    if not loaded:
        raise FileNotFoundError(f"No shoulder bone masks found in {seg_dir}")
    return combined, loaded


def save_combined_shoulder_bone_mask(
    seg_dir: str | Path,
    output_path: str | Path,
    reference: NiftiImage,
) -> tuple[np.ndarray, list[str]]:
    mask, loaded = load_totalseg_shoulder_bone_mask(seg_dir, reference.data.shape)
    save_nifti_like(output_path, mask.astype(np.uint8), reference=reference)
    return mask, loaded
