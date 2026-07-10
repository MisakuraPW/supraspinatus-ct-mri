from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], continue_on_error: bool) -> int:
    print("\n==>", " ".join(cmd))
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0 and not continue_on_error:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-training TotalSeg bone segmentation improvement suite.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--output-root", default="outputs/2026-07_totalseg_bone_segmentation_suite")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu", "mps"))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-fast", action="store_true", help="Run TotalSeg fast model.")
    parser.add_argument("--run-fullres", action="store_true", help="Run TotalSeg without --fast.")
    parser.add_argument("--skip-totalseg", action="store_true", help="Only run fusion using existing TotalSeg outputs.")
    parser.add_argument("--fast-dir", default=None, help="Existing or target fast TotalSeg output dir.")
    parser.add_argument("--fullres-dir", default=None, help="Existing or target full-resolution TotalSeg output dir.")
    parser.add_argument("--fusion-dir", default=None)
    parser.add_argument("--guided-dilation-voxels", type=int, default=4)
    parser.add_argument("--min-totalseg-voxels", type=int, default=10000)
    parser.add_argument("--min-fused-voxels", type=int, default=10000)
    args = parser.parse_args()

    if not args.run_fast and not args.run_fullres and not args.skip_totalseg:
        args.run_fast = True
        args.run_fullres = True

    output_root = Path(args.output_root)
    fast_dir = Path(args.fast_dir) if args.fast_dir else output_root / "totalseg_fast"
    fullres_dir = Path(args.fullres_dir) if args.fullres_dir else output_root / "totalseg_fullres"
    fusion_dir = Path(args.fusion_dir) if args.fusion_dir else output_root / "totalseg_hu_fused"

    mask_dirs: list[Path] = []
    common_case_args: list[str] = []
    if args.cases:
        common_case_args = ["--cases", *args.cases]

    if args.run_fast or (args.skip_totalseg and fast_dir.exists()):
        mask_dirs.append(fast_dir)
    if args.run_fullres or (args.skip_totalseg and fullres_dir.exists()):
        mask_dirs.append(fullres_dir)

    if not args.skip_totalseg and args.run_fast:
        cmd = [
            sys.executable,
            "scripts/run_totalseg_shoulder_bones.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(fast_dir),
            "--device",
            args.device,
            "--fast",
            *common_case_args,
        ]
        if args.require_gpu:
            cmd.append("--require-gpu")
        if args.skip_existing:
            cmd.append("--skip-existing")
        run_command(cmd, args.continue_on_error)

    if not args.skip_totalseg and args.run_fullres:
        cmd = [
            sys.executable,
            "scripts/run_totalseg_shoulder_bones.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(fullres_dir),
            "--device",
            args.device,
            *common_case_args,
        ]
        if args.require_gpu:
            cmd.append("--require-gpu")
        if args.skip_existing:
            cmd.append("--skip-existing")
        run_command(cmd, args.continue_on_error)

    existing_mask_dirs = [path for path in mask_dirs if path.exists()]
    if not existing_mask_dirs:
        raise SystemExit("No TotalSeg mask dirs are available. Run TotalSeg first or pass --fast-dir/--fullres-dir with --skip-totalseg.")

    fusion_cmd = [
        sys.executable,
        "scripts/fuse_totalseg_hu_bone_masks.py",
        "--data-dir",
        args.data_dir,
        "--totalseg-mask-dirs",
        *[str(path) for path in existing_mask_dirs],
        "--output-dir",
        str(fusion_dir),
        "--guided-dilation-voxels",
        str(args.guided_dilation_voxels),
        "--min-totalseg-voxels",
        str(args.min_totalseg_voxels),
        "--min-fused-voxels",
        str(args.min_fused_voxels),
        *common_case_args,
    ]
    run_command(fusion_cmd, args.continue_on_error)

    print("\nRecommended next locator command:")
    print(
        "python scripts/run_totalseg_locator_strategy_sweep.py "
        f"--data-dir {args.data_dir} "
        f"--bone-mask-dir {fusion_dir} "
        "--bone-mask-filename shoulder_bones_fused_hu.nii.gz "
        "--output-root outputs/2026-07_totalseg_fused_locator_sweep "
        "--quick"
    )


if __name__ == "__main__":
    main()
