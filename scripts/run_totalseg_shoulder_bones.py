from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.preprocessing.totalseg_bones import SHOULDER_BONE_CLASSES, save_combined_shoulder_bone_mask


def find_ct_60kev(case_dir: Path) -> Path:
    ct_dir = case_dir / "CT"
    matches = sorted(path for path in ct_dir.iterdir() if path.is_file() and "60" in path.name.lower() and ".nii" in path.name.lower())
    if not matches:
        raise FileNotFoundError(f"No 60keV NIfTI found under {ct_dir}")
    return matches[0]


def discover_cases(data_dir: Path, case_names: list[str] | None) -> list[Path]:
    cases = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if case_names:
        wanted = set(case_names)
        cases = [path for path in cases if path.name in wanted]
    return cases


def resolve_totalseg_command(command: str) -> list[str]:
    if command == "auto":
        exe = shutil.which("TotalSegmentator")
        if exe:
            return [exe]
        local_cli = Path("refercode") / "TotalSegmentator" / "totalsegmentator" / "bin" / "TotalSegmentator.py"
        if local_cli.exists():
            return [sys.executable, str(local_cli)]
        return ["TotalSegmentator"]
    parts = command.split()
    return parts if parts else ["TotalSegmentator"]


def build_command(args: argparse.Namespace, ct_path: Path, seg_dir: Path, report_path: Path) -> list[str]:
    cmd = resolve_totalseg_command(args.totalseg_command)
    cmd.extend(
        [
            "-i",
            str(ct_path),
            "-o",
            str(seg_dir),
            "-ta",
            "total",
            "--roi_subset",
            *SHOULDER_BONE_CLASSES,
            "--report",
            str(report_path),
            "-d",
            args.device,
        ]
    )
    if args.fast:
        cmd.append("--fast")
    if args.statistics:
        cmd.extend(["--statistics", "--statistics_extra"])
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def run_case(args: argparse.Namespace, case_dir: Path, output_dir: Path) -> dict[str, object]:
    case_out = output_dir / case_dir.name
    seg_dir = case_out / "segmentations"
    report_path = case_out / "run_report.json"
    combined_path = case_out / "shoulder_bones_combined.nii.gz"
    case_out.mkdir(parents=True, exist_ok=True)
    ct_path = find_ct_60kev(case_dir)
    cmd = build_command(args, ct_path, seg_dir, report_path)
    row: dict[str, object] = {
        "case": case_dir.name,
        "ct_path": str(ct_path),
        "seg_dir": str(seg_dir),
        "combined_mask": str(combined_path),
        "command": " ".join(cmd),
        "returncode": "",
        "status": "pending",
        "loaded_classes": "",
        "combined_voxels": "",
        "report_path": str(report_path),
    }
    if args.dry_run:
        row["status"] = "dry_run"
        return row
    if args.skip_existing and combined_path.exists():
        image = load_nifti(ct_path)
        mask, loaded = save_combined_shoulder_bone_mask(seg_dir, combined_path, image)
        row.update({"status": "skipped_existing", "loaded_classes": " ".join(loaded), "combined_voxels": int(mask.sum()), "returncode": 0})
        return row

    env = os.environ.copy()
    local_source = Path("refercode") / "TotalSegmentator"
    if local_source.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(local_source) if not existing else str(local_source) + os.pathsep + existing
    result = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, capture_output=True, env=env)
    (case_out / "stdout.txt").write_text(result.stdout, encoding="utf-8", errors="replace")
    (case_out / "stderr.txt").write_text(result.stderr, encoding="utf-8", errors="replace")
    row["returncode"] = result.returncode
    if result.returncode != 0:
        row["status"] = "failed"
        return row

    image = load_nifti(ct_path)
    mask, loaded = save_combined_shoulder_bone_mask(seg_dir, combined_path, image)
    row.update({"status": "ok", "loaded_classes": " ".join(loaded), "combined_voxels": int(mask.sum())})
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row["runtime_seconds"] = report.get("runtime_seconds", "")
            row["device"] = report.get("device", "")
        except Exception:
            pass
    return row


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "case",
        "status",
        "returncode",
        "combined_voxels",
        "loaded_classes",
        "runtime_seconds",
        "device",
        "ct_path",
        "seg_dir",
        "combined_mask",
        "report_path",
        "command",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TotalSegmentator shoulder-bone masks for project CT volumes.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--output-dir", default="outputs/2026-07_totalseg_shoulder_bones")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu", "mps"))
    parser.add_argument("--fast", action="store_true", help="Use lower-resolution TotalSegmentator model.")
    parser.add_argument("--statistics", action="store_true", help="Request statistics.json and statistics_extra.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--totalseg-command", default="auto", help="Command prefix, or 'auto' for installed/local CLI.")
    args = parser.parse_args()

    rows = []
    for case_dir in discover_cases(Path(args.data_dir), args.cases):
        print(f"TotalSegmentator shoulder bones: {case_dir.name}")
        row = run_case(args, case_dir, Path(args.output_dir))
        rows.append(row)
        print(f"  {row['status']} returncode={row['returncode']} combined={row['combined_voxels']}")
    write_summary(rows, Path(args.output_dir) / "totalseg_shoulder_bones_summary.csv")
    print(f"wrote summary to {Path(args.output_dir) / 'totalseg_shoulder_bones_summary.csv'}")


if __name__ == "__main__":
    main()
