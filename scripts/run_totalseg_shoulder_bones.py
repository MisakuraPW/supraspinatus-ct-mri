from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.preprocessing.totalseg_bones import SHOULDER_BONE_CLASSES, save_combined_shoulder_bone_mask


def get_torch_device_info() -> dict[str, object]:
    info: dict[str, object] = {
        "torch_available": False,
        "torch_version": "",
        "torch_cuda_build": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": "",
        "error": "",
    }
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda or ""
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def classify_failure(stdout: str, stderr: str) -> tuple[str, str]:
    combined = f"{stdout}\n{stderr}".lower()
    hints: list[str] = []
    if "chunkedencodingerror" in combined or "incompleteread" in combined or "connection broken" in combined:
        hints.append("weight_download_interrupted")
    if "no gpu detected" in combined or "cuda initialization" in combined or "driver on your system is too old" in combined:
        hints.append("gpu_unavailable_or_torch_cuda_mismatch")
    if "no space left on device" in combined:
        hints.append("disk_full")
    if "modulenotfounderror" in combined or "command not found" in combined:
        hints.append("totalsegmentator_install_or_path_error")
    if not hints:
        hints.append("see_stdout_stderr")
    return "failed_" + "_".join(hints[:2]), ";".join(hints)


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


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.totalseg_home_dir:
        env["TOTALSEG_HOME_DIR"] = str(Path(args.totalseg_home_dir))
    local_source = Path("refercode") / "TotalSegmentator"
    if local_source.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(local_source) if not existing else str(local_source) + os.pathsep + existing
    return env


def resolve_download_command(command: str) -> list[str]:
    if command != "auto":
        parts = command.split()
        return parts if parts else ["totalseg_download_weights"]
    exe = shutil.which("totalseg_download_weights")
    if exe:
        return [exe]
    local_cli = Path("refercode") / "TotalSegmentator" / "totalsegmentator" / "bin" / "totalseg_download_weights.py"
    if local_cli.exists():
        return [sys.executable, str(local_cli)]
    return ["totalseg_download_weights"]


def download_weights_with_retries(args: argparse.Namespace) -> None:
    task_name = args.weight_task or ("total_fast" if args.fast else "total")
    base_cmd = resolve_download_command(args.totalseg_download_command)
    cmd = [*base_cmd, "-t", task_name]
    env = build_env(args)
    last_returncode = 0
    for attempt in range(1, int(args.weight_download_retries) + 1):
        print(f"pre-download TotalSegmentator weights: task={task_name} attempt={attempt}/{args.weight_download_retries}")
        result = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, env=env)
        last_returncode = int(result.returncode)
        if result.returncode == 0:
            print("  weight pre-download ok")
            return
        print(f"  weight pre-download failed returncode={result.returncode}")
        if attempt < int(args.weight_download_retries):
            time.sleep(float(args.retry_delay_seconds))
    raise SystemExit(f"ERROR: TotalSegmentator weight pre-download failed after {args.weight_download_retries} attempts (returncode={last_returncode}).")


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
        "failure_hint": "",
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

    env = build_env(args)
    result: subprocess.CompletedProcess[str] | None = None
    status = ""
    hint = ""
    for attempt in range(1, int(args.retries) + 1):
        result = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, capture_output=True, env=env)
        (case_out / f"stdout_attempt{attempt}.txt").write_text(result.stdout, encoding="utf-8", errors="replace")
        (case_out / f"stderr_attempt{attempt}.txt").write_text(result.stderr, encoding="utf-8", errors="replace")
        (case_out / "stdout.txt").write_text(result.stdout, encoding="utf-8", errors="replace")
        (case_out / "stderr.txt").write_text(result.stderr, encoding="utf-8", errors="replace")
        row["returncode"] = result.returncode
        row["attempts"] = attempt
        if result.returncode == 0:
            break
        status, hint = classify_failure(result.stdout, result.stderr)
        row["status"] = status
        row["failure_hint"] = hint
        retryable = "weight_download_interrupted" in hint
        if not retryable or attempt >= int(args.retries):
            return row
        print(f"  retrying {case_dir.name} after {hint}; attempt {attempt + 1}/{args.retries}")
        time.sleep(float(args.retry_delay_seconds))
    if result is None:
        row["status"] = "failed_no_subprocess_result"
        row["failure_hint"] = "internal_error"
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
        "failure_hint",
        "combined_voxels",
        "loaded_classes",
        "runtime_seconds",
        "device",
        "ct_path",
        "seg_dir",
        "combined_mask",
        "report_path",
        "command",
        "attempts",
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
    parser.add_argument("--require-gpu", action="store_true", help="Fail before running cases if --device gpu is requested but torch CUDA is unavailable.")
    parser.add_argument("--fast", action="store_true", help="Use lower-resolution TotalSegmentator model.")
    parser.add_argument("--statistics", action="store_true", help="Request statistics.json and statistics_extra.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop the batch immediately when one case fails.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--totalseg-command", default="auto", help="Command prefix, or 'auto' for installed/local CLI.")
    parser.add_argument("--totalseg-home-dir", default=None, help="Set TOTALSEG_HOME_DIR, e.g. outputs/.../.totalsegmentator.")
    parser.add_argument("--retries", type=int, default=1, help="Retry a case when weight download is interrupted.")
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--download-weights-first", action="store_true", help="Run totalseg_download_weights before case inference.")
    parser.add_argument("--weight-task", default=None, help="Override download task; defaults to total_fast for --fast, otherwise total.")
    parser.add_argument("--weight-download-retries", type=int, default=3)
    parser.add_argument("--totalseg-download-command", default="auto", help="Command prefix for totalseg_download_weights.")
    args = parser.parse_args()

    if args.device == "gpu":
        device_info = get_torch_device_info()
        print(
            "torch device check: "
            f"torch={device_info['torch_version'] or 'unavailable'} "
            f"cuda_build={device_info['torch_cuda_build'] or 'none'} "
            f"cuda_available={device_info['cuda_available']} "
            f"gpu={device_info['cuda_device_name'] or 'none'}"
        )
        if args.require_gpu and not device_info["cuda_available"]:
            raise SystemExit(
                "ERROR: --require-gpu was set, but torch.cuda.is_available() is False. "
                "Fix the cloud CUDA/PyTorch environment or run with --device cpu."
            )

    if args.download_weights_first and not args.dry_run:
        download_weights_with_retries(args)

    rows = []
    for case_dir in discover_cases(Path(args.data_dir), args.cases):
        print(f"TotalSegmentator shoulder bones: {case_dir.name}")
        row = run_case(args, case_dir, Path(args.output_dir))
        rows.append(row)
        print(f"  {row['status']} returncode={row['returncode']} combined={row['combined_voxels']}")
        if row.get("failure_hint"):
            print(f"  hint={row['failure_hint']}")
        if args.stop_on_fail and str(row["status"]).startswith("failed"):
            print("stopped on first failure because --stop-on-fail was set")
            break
    write_summary(rows, Path(args.output_dir) / "totalseg_shoulder_bones_summary.csv")
    print(f"wrote summary to {Path(args.output_dir) / 'totalseg_shoulder_bones_summary.csv'}")


if __name__ == "__main__":
    main()
