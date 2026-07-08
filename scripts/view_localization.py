from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.viewer.volume_viewer import ViewerData, run_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="View image, manual ROI, and predicted localization overlay.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    image = load_nifti(args.image).data
    mask = load_nifti(args.mask).data if args.mask else None
    pred = load_nifti(args.pred).data
    raise SystemExit(run_viewer(ViewerData(image=image, mask=mask, pred=pred)))


if __name__ == "__main__":
    main()

