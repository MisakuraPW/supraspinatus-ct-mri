from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.localization.multi_bone_traditional import MultiBoneConfig, process_dataset


FOCUS_CASES = ("SB", "WQX", "ZJ", "OSQ")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(run_id: str, params: dict[str, object], run_dir: Path) -> dict[str, object]:
    summary = read_csv(run_dir / "results" / "summary_metrics.csv")[0]
    final_rows = read_csv(run_dir / "results" / "per_case_final.csv")
    failure_rows = {row["case"]: row for row in read_csv(run_dir / "results" / "failure_analysis.csv")}
    row: dict[str, object] = {
        "run_id": run_id,
        **params,
        "mean_center_error": summary["mean_center_error_mm"],
        "median_center_error": summary["median_center_error_mm"],
        "worst_case_center_error": summary["worst_center_error_mm"],
        "mean_bbox_iou": summary["mean_bbox_iou"],
        "mean_doctor_roi_coverage": summary["mean_doctor_roi_coverage"],
        "top3_best_coverage": summary["top3_best_coverage_mean"],
        "top5_best_coverage": summary["top5_best_coverage_mean"],
        "mean_bone_overlap": summary["mean_pred_bone_overlap"],
        "generation_failure_count": summary["generation_failure_count"],
        "ranking_failure_count": summary["ranking_failure_count"],
    }
    final_by_case = {case_row["case"]: case_row for case_row in final_rows}
    for case in FOCUS_CASES:
        if case in final_by_case:
            row[f"{case}_center_error"] = final_by_case[case]["center_error_mm"]
            row[f"{case}_coverage"] = final_by_case[case]["doctor_roi_coverage"]
            row[f"{case}_bone_overlap"] = final_by_case[case]["pred_bone_overlap"]
        if case in failure_rows:
            row[f"{case}_failure_type"] = failure_rows[case]["failure_type"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search traditional multi-bone locator parameters.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--output-dir", default="outputs/multibone_grid_search")
    parser.add_argument("--teacher-csv", default="outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all preset profiles.")
    parser.add_argument("--full-grid", action="store_true", help="Run the expensive full Cartesian grid.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.full_grid:
        import itertools

        grid = {
            "profile": ["full_grid"],
            "bone_margin_voxels": [1, 2, 3],
            "continuity_window": [1, 2, 3],
            "continuity_xy_tolerance": [10.0, 14.0, 18.0],
            "bone_dist_good_min_mm": [2.0, 3.0, 4.0],
            "bone_dist_good_max_mm": [6.0, 8.0, 10.0],
            "low_z_range_mm": [8.0, 12.0, 16.0],
            "low_z_step_mm": [2.0],
            "teacher_z_window_mm": [6.0, 8.0, 10.0],
            "teacher_z_refine_step_mm": [2.0],
            "bone_dist_band_weight": [1.0],
        }
        keys = list(grid)
        param_sets = [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]
    else:
        param_sets = [
            {
                "profile": "current_best",
                "bone_margin_voxels": 2,
                "continuity_window": 2,
                "continuity_xy_tolerance": 14.0,
                "bone_dist_good_min_mm": 3.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 12.0,
                "low_z_step_mm": 2.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 1.0,
            },
            {
                "profile": "coarse_low_z_wide",
                "bone_margin_voxels": 2,
                "continuity_window": 2,
                "continuity_xy_tolerance": 14.0,
                "bone_dist_good_min_mm": 3.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 18.0,
                "low_z_step_mm": 4.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 1.0,
            },
            {
                "profile": "low_z_mid_refine",
                "bone_margin_voxels": 2,
                "continuity_window": 2,
                "continuity_xy_tolerance": 14.0,
                "bone_dist_good_min_mm": 3.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 16.0,
                "low_z_step_mm": 3.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 1.0,
            },
            {
                "profile": "teacher_z_wide",
                "bone_margin_voxels": 2,
                "continuity_window": 2,
                "continuity_xy_tolerance": 14.0,
                "bone_dist_good_min_mm": 3.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 12.0,
                "low_z_step_mm": 2.0,
                "teacher_z_window_mm": 12.0,
                "teacher_z_refine_step_mm": 3.0,
                "bone_dist_band_weight": 1.0,
            },
            {
                "profile": "bone_tolerant",
                "bone_margin_voxels": 1,
                "continuity_window": 2,
                "continuity_xy_tolerance": 14.0,
                "bone_dist_good_min_mm": 2.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 12.0,
                "low_z_step_mm": 2.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 0.8,
            },
            {
                "profile": "bone_conservative",
                "bone_margin_voxels": 3,
                "continuity_window": 2,
                "continuity_xy_tolerance": 14.0,
                "bone_dist_good_min_mm": 4.0,
                "bone_dist_good_max_mm": 10.0,
                "low_z_range_mm": 12.0,
                "low_z_step_mm": 2.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 1.2,
            },
            {
                "profile": "continuity_strict",
                "bone_margin_voxels": 2,
                "continuity_window": 3,
                "continuity_xy_tolerance": 10.0,
                "bone_dist_good_min_mm": 3.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 12.0,
                "low_z_step_mm": 2.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 1.0,
            },
            {
                "profile": "continuity_loose",
                "bone_margin_voxels": 2,
                "continuity_window": 1,
                "continuity_xy_tolerance": 18.0,
                "bone_dist_good_min_mm": 3.0,
                "bone_dist_good_max_mm": 8.0,
                "low_z_range_mm": 12.0,
                "low_z_step_mm": 2.0,
                "teacher_z_window_mm": 8.0,
                "teacher_z_refine_step_mm": 2.0,
                "bone_dist_band_weight": 1.0,
            },
        ]
    rows: list[dict[str, object]] = []
    for index, params in enumerate(param_sets, start=1):
        if args.max_runs and index > args.max_runs:
            break
        run_id = f"run_{index:04d}"
        run_dir = output_dir / "runs" / run_id
        cfg = MultiBoneConfig(
            bone_margin_voxels=int(params["bone_margin_voxels"]),
            continuity_window=int(params["continuity_window"]),
            continuity_xy_tolerance=float(params["continuity_xy_tolerance"]),
            top_k=args.topk,
            low_z_enable=True,
            low_z_range_mm=float(params["low_z_range_mm"]),
            low_z_step_mm=float(params["low_z_step_mm"]),
            teacher_z_refine_enable=True,
            teacher_z_window_mm=float(params["teacher_z_window_mm"]),
            teacher_z_refine_step_mm=float(params["teacher_z_refine_step_mm"]),
            bone_dist_good_min_mm=float(params["bone_dist_good_min_mm"]),
            bone_dist_good_max_mm=float(params["bone_dist_good_max_mm"]),
            bone_dist_band_weight=float(params["bone_dist_band_weight"]),
        )
        process_dataset(args.data_dir, run_dir, cfg, teacher_csv=args.teacher_csv)
        rows.append(summarize_run(run_id, params, run_dir))
        write_csv(output_dir / "grid_search_summary.csv", rows)

    write_csv(output_dir / "grid_search_summary.csv", rows)
    write_csv(output_dir / "grid_search_best_by_mean.csv", sorted(rows, key=lambda row: float(row["mean_center_error"]))[:10])
    write_csv(output_dir / "grid_search_best_by_worstcase.csv", sorted(rows, key=lambda row: float(row["worst_case_center_error"]))[:10])
    write_csv(output_dir / "grid_search_best_by_coverage.csv", sorted(rows, key=lambda row: float(row["mean_doctor_roi_coverage"]), reverse=True)[:10])
    print(f"wrote grid search outputs to {output_dir}")


if __name__ == "__main__":
    main()
