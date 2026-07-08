from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti, save_nifti_like
from supraspinatus_locator.localization.centerline import centerline_heatmap_from_bbox
from supraspinatus_locator.localization.roi_geometry import BBox3D


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a centerline heatmap from a rule-based ROI json.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--roi-json", required=True)
    parser.add_argument("--out", default="outputs/localization/centerline_heatmap.nii.gz")
    parser.add_argument("--sigma-voxels", type=float, default=3.0)
    args = parser.parse_args()

    image = load_nifti(args.image)
    payload = json.loads(Path(args.roi_json).read_text(encoding="utf-8"))
    bbox_payload = payload["bbox"]
    bbox = BBox3D(tuple(bbox_payload["min"]), tuple(bbox_payload["max"]))
    heatmap = centerline_heatmap_from_bbox(image.data.shape, bbox, args.sigma_voxels)
    save_nifti_like(args.out, heatmap.astype("float32"), reference=image)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

