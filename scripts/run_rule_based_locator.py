from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti, save_nifti_like
from supraspinatus_locator.localization.rule_based import RuleBasedConfig, locate_roi, save_localization_outputs
from supraspinatus_locator.localization.visualize_localization import save_overlay_montage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based supraspinatus candidate ROI localization.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-mask")
    parser.add_argument("--out-dir", default="outputs/localization/rule_based")
    parser.add_argument("--bone-threshold", type=float, default=180.0)
    parser.add_argument("--min-bone-voxels", type=int, default=5000)
    parser.add_argument("--roi-size-mm", nargs=3, type=float, default=(34.0, 18.0, 12.0))
    parser.add_argument("--roi-offset-mm", nargs=3, type=float, default=(0.0, -6.0, 0.0))
    parser.add_argument("--center-strategy", choices=["bone_bbox_fraction", "bone_centroid_offset"], default="bone_bbox_fraction")
    parser.add_argument("--bone-bbox-fraction", nargs=3, type=float, default=(0.84, 0.10, 0.52))
    args = parser.parse_args()

    image = load_nifti(args.image)
    target = load_nifti(args.target_mask).data if args.target_mask else None
    config = RuleBasedConfig(
        bone_threshold_hu=args.bone_threshold,
        min_bone_component_voxels=args.min_bone_voxels,
        roi_size_mm=tuple(args.roi_size_mm),
        roi_offset_from_bone_center_mm=tuple(args.roi_offset_mm),
        center_strategy=args.center_strategy,
        bone_bbox_fraction_xyz=tuple(args.bone_bbox_fraction),
    )
    roi, bbox, meta = locate_roi(image.data, tuple(image.spacing[:3]), config)
    out = Path(args.out_dir)
    save_nifti_like(out / "roi_mask.nii.gz", roi.astype("uint8"), reference=image)
    payload = save_localization_outputs(out, roi, bbox, meta, target)
    masks = [(roi, (64, 220, 255))]
    if target is not None:
        masks.append((target, (255, 64, 64)))
    save_overlay_montage(image.data, out / "preview.png", masks=masks, center=(bbox.min[2] + bbox.max[2]) // 2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
