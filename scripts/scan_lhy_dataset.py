from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.dataset_scan import scan_dataset, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan LHY data files and write a JSON summary.")
    parser.add_argument("--root", default="LHY/LHY")
    parser.add_argument("--out", default="outputs/summaries/lhy_summary.json")
    args = parser.parse_args()
    summary = scan_dataset(args.root)
    write_summary(summary, args.out)
    print(f"wrote {args.out}")
    print(f"nifti={len(summary['nifti'])} dicom_series={len(summary['dicom_series'])}")


if __name__ == "__main__":
    main()

