from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.viewer.volume_viewer import ViewerData, run_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the interactive volume viewer.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--pred")
    parser.add_argument("--window-center", type=float, default=80.0)
    parser.add_argument("--window-width", type=float, default=500.0)
    args = parser.parse_args()
    image = load_nifti(args.image).data
    mask = load_nifti(args.mask).data if args.mask else None
    pred = load_nifti(args.pred).data if args.pred else None
    raise SystemExit(run_viewer(ViewerData(image=image, mask=mask, pred=pred, window_center=args.window_center, window_width=args.window_width)))


if __name__ == "__main__":
    main()

