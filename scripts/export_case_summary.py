from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.localization.roi_geometry import bbox_from_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compact one-case summary for reports.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--out", default="outputs/summaries/case_summary.json")
    args = parser.parse_args()
    image = load_nifti(args.image)
    summary = {
        "image": args.image,
        "shape": list(image.data.shape),
        "spacing": list(image.spacing),
        "min": float(image.data.min()),
        "max": float(image.data.max()),
    }
    if args.mask:
        mask = load_nifti(args.mask).data
        bbox = bbox_from_mask(mask > 0)
        summary["mask"] = {"path": args.mask, "voxels": int((mask > 0).sum()), "bbox": bbox.to_dict() if bbox else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

