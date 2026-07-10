from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def build_experiments(quick: bool) -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = [
        {"name": "00_threshold_roi_generalized", "policy": "generalized", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "01_threshold_roi_best_score", "policy": "best_score", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "02_threshold_roi_current_first", "policy": "current_first", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "03_threshold_roi_conservative", "policy": "conservative", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "04_threshold_roi_consensus", "policy": "consensus", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "05_threshold_roi_edge_priority", "policy": "edge_priority", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "06_threshold_roi_surface_suppressed", "policy": "surface_suppressed", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "07_threshold_roi_adaptive_edge_guarded", "policy": "adaptive_edge_guarded", "mode": "threshold_roi", "dilation": 4, "bone": True},
        {"name": "08_hu_conservative_baseline", "policy": "conservative", "mode": "", "dilation": 0, "bone": False},
        {"name": "09_hu_adaptive_edge_guarded", "policy": "adaptive_edge_guarded", "mode": "", "dilation": 0, "bone": False},
        {"name": "10_threshold_roi_oracle_upper_bound", "policy": "oracle", "mode": "threshold_roi", "dilation": 4, "bone": True},
    ]
    if not quick:
        experiments.extend(
            [
                {"name": "11_threshold_roi_conservative_dilate2", "policy": "conservative", "mode": "threshold_roi", "dilation": 2, "bone": True},
                {"name": "12_threshold_roi_conservative_dilate8", "policy": "conservative", "mode": "threshold_roi", "dilation": 8, "bone": True},
                {"name": "13_threshold_roi_consensus_dilate8", "policy": "consensus", "mode": "threshold_roi", "dilation": 8, "bone": True},
                {"name": "14_threshold_roi_adaptive_edge_guarded_dilate8", "policy": "adaptive_edge_guarded", "mode": "threshold_roi", "dilation": 8, "bone": True},
                {"name": "15_direct_conservative_probe", "policy": "conservative", "mode": "direct", "dilation": 0, "bone": True},
            ]
        )
    return experiments


def read_first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TotalSeg-guided locator strategy sweep.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--bone-mask-dir", default="outputs/2026-07_totalseg_shoulder_bones")
    parser.add_argument("--output-root", default="outputs/2026-07_totalseg_locator_strategy_sweep")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="Run core policies only.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-ranker", action="store_true", help="Skip candidate-level LOOCV ranker experiments.")
    parser.add_argument("--candidate-preview-topk", type=int, default=8)
    parser.add_argument("--min-external-bone-voxels", type=int, default=10000)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(args.quick)
    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    per_case_rows: list[dict[str, object]] = []
    ranker_summary_rows: list[dict[str, object]] = []

    for exp in experiments:
        exp_dir = output_root / str(exp["name"])
        cmd = [
            sys.executable,
            "scripts/run_multibone_locator.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(exp_dir),
            "--selection-policy",
            str(exp["policy"]),
            "--surface-arc-enable",
            "--bone-edge-enable",
            "--export-candidates",
            "--candidate-preview-topk",
            str(args.candidate_preview_topk),
        ]
        if args.cases:
            cmd.extend(["--cases", *args.cases])
        if exp["bone"]:
            cmd.extend(
                [
                    "--bone-mask-dir",
                    args.bone_mask_dir,
                    "--allow-threshold-bone-fallback",
                    "--min-external-bone-voxels",
                    str(args.min_external_bone_voxels),
                    "--external-bone-mode",
                    str(exp["mode"]),
                ]
            )
            if str(exp["mode"]) == "threshold_roi":
                cmd.extend(["--external-bone-dilation-voxels", str(exp["dilation"])])
        print("\n==>", exp["name"])
        print(" ".join(cmd))
        proc = subprocess.run(cmd, text=True)
        manifest_rows.append(
            {
                **exp,
                "output_dir": str(exp_dir),
                "returncode": proc.returncode,
                "command": " ".join(cmd),
            }
        )
        if proc.returncode != 0 and not args.continue_on_error:
            write_csv(output_root / "experiment_manifest.csv", manifest_rows)
            raise SystemExit(proc.returncode)

        summary = read_first_csv_row(exp_dir / "results" / "summary_metrics.csv")
        if summary:
            summary_rows.append({**exp, **summary, "output_dir": str(exp_dir), "returncode": proc.returncode})
        for row in read_csv_rows(exp_dir / "results" / "per_case_final.csv"):
            per_case_rows.append({**exp, **row, "experiment": str(exp["name"]), "output_dir": str(exp_dir)})
        if proc.returncode == 0 and not args.skip_ranker:
            ranker_dir = exp_dir / "ranker"
            ranker_cmd = [
                sys.executable,
                "scripts/run_candidate_ranker_experiment.py",
                "--labeled-candidates",
                str(exp_dir / "results" / "per_case_topk.csv"),
                "--unlabeled-candidates",
                str(exp_dir / "results" / "_missing_unlabeled_candidates.csv"),
                "--unlabeled-feedback",
                str(exp_dir / "results" / "_missing_unlabeled_feedback.csv"),
                "--output-dir",
                str(ranker_dir),
            ]
            print("ranker:", " ".join(ranker_cmd))
            ranker_proc = subprocess.run(ranker_cmd, text=True)
            manifest_rows.append(
                {
                    **exp,
                    "output_dir": str(ranker_dir),
                    "returncode": ranker_proc.returncode,
                    "command": " ".join(ranker_cmd),
                    "stage": "ranker",
                }
            )
            if ranker_proc.returncode != 0 and not args.continue_on_error:
                write_csv(output_root / "experiment_manifest.csv", manifest_rows)
                raise SystemExit(ranker_proc.returncode)
            for row in read_csv_rows(ranker_dir / "results" / "policy_summary.csv"):
                ranker_summary_rows.append({**exp, **row, "experiment": str(exp["name"]), "ranker_output_dir": str(ranker_dir)})

    write_csv(output_root / "experiment_manifest.csv", manifest_rows)
    write_csv(output_root / "combined_summary_metrics.csv", summary_rows)
    write_csv(output_root / "combined_per_case_final.csv", per_case_rows)
    write_csv(output_root / "combined_ranker_policy_summary.csv", ranker_summary_rows)

    if summary_rows:
        ranked = sorted(summary_rows, key=lambda row: float(row.get("mean_center_error_mm", 9999)))
        write_csv(output_root / "ranked_summary_metrics.csv", ranked)
        print("\nBest by mean_center_error_mm:")
        for row in ranked[:5]:
            print(
                f"{row['name']}: mean={row.get('mean_center_error_mm')} "
                f"worst={row.get('worst_center_error_mm')} cov={row.get('mean_doctor_roi_coverage')} "
                f"iou={row.get('mean_bbox_iou')} bone={row.get('mean_pred_bone_overlap')}"
            )
    print(f"\nwrote sweep outputs to {output_root}")


if __name__ == "__main__":
    main()
