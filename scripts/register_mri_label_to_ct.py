from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.localization.registration_weak_label import rigid_register_mri_label_to_ct


def main() -> None:
    parser = argparse.ArgumentParser(description="Route D: rigidly register an MRI label into CT space with SimpleITK.")
    parser.add_argument("--ct-image", required=True)
    parser.add_argument("--mri-image", required=True)
    parser.add_argument("--mri-label", required=True)
    parser.add_argument("--out-label", required=True)
    args = parser.parse_args()
    rigid_register_mri_label_to_ct(args.ct_image, args.mri_image, args.mri_label, args.out_label)
    print(f"wrote {args.out_label}")


if __name__ == "__main__":
    main()

