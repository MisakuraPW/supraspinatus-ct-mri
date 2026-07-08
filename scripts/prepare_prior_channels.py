from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti, save_nifti_like
from supraspinatus_locator.preprocessing.body_bone_masks import threshold_bone
from supraspinatus_locator.preprocessing.distance_maps import build_prior_channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bone/distance/coordinate prior channels for route C.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="outputs/priors")
    parser.add_argument("--bone-threshold", type=float, default=180.0)
    args = parser.parse_args()

    image = load_nifti(args.image)
    bone = threshold_bone(image.data, args.bone_threshold)
    channels = build_prior_channels(bone, tuple(image.spacing[:3]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["bone_mask", "bone_distance", "coord_x", "coord_y", "coord_z"]
    for idx, name in enumerate(names[: channels.shape[0]]):
        save_nifti_like(out_dir / f"{name}.nii.gz", channels[idx].astype("float32"), reference=image)
    print(f"wrote {channels.shape[0]} prior channels to {out_dir}")


if __name__ == "__main__":
    main()

