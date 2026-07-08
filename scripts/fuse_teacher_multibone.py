from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "case",
    "selected_method",
    "selection_reason",
    "center_error_mm",
    "pred_box_doctor_bbox_iou",
    "doctor_roi_coverage",
    "pred_bone_overlap",
    "teacher_center_error_mm",
    "multibone_center_error_mm",
    "teacher_doctor_roi_coverage",
    "multibone_doctor_roi_coverage",
    "teacher_bone_overlap",
    "multibone_bone_overlap",
    "multibone_score",
]


def read_csv_by_case(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def select_method(teacher: dict[str, str], multibone: dict[str, str], score_threshold: float) -> tuple[str, str, dict[str, str]]:
    multibone_score = as_float(multibone, "score")
    if multibone_score < score_threshold:
        return "teacher_single_humerus_anchor", f"multibone_score<{score_threshold:g}", teacher
    return "multi_bone_roof_humerus", f"multibone_score>={score_threshold:g}", multibone


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse teacher single-anchor and multi-bone traditional locator results.")
    parser.add_argument("--teacher-csv", default="outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv")
    parser.add_argument("--multibone-csv", default="outputs/multibone_10cases/multibone_locator_results.csv")
    parser.add_argument("--score-threshold", type=float, default=3.7)
    parser.add_argument("--out", default="outputs/hybrid_10cases/hybrid_teacher_multibone_results.csv")
    args = parser.parse_args()

    teacher_rows = read_csv_by_case(Path(args.teacher_csv))
    multibone_rows = read_csv_by_case(Path(args.multibone_csv))
    rows: list[dict[str, object]] = []

    for case in sorted(set(teacher_rows) & set(multibone_rows)):
        teacher = teacher_rows[case]
        multibone = multibone_rows[case]
        selected_method, reason, selected = select_method(teacher, multibone, args.score_threshold)
        rows.append(
            {
                "case": case,
                "selected_method": selected_method,
                "selection_reason": reason,
                "center_error_mm": as_float(selected, "center_error_mm"),
                "pred_box_doctor_bbox_iou": as_float(selected, "pred_box_doctor_bbox_iou"),
                "doctor_roi_coverage": as_float(selected, "doctor_roi_coverage"),
                "pred_bone_overlap": as_float(selected, "pred_bone_overlap"),
                "teacher_center_error_mm": as_float(teacher, "center_error_mm"),
                "multibone_center_error_mm": as_float(multibone, "center_error_mm"),
                "teacher_doctor_roi_coverage": as_float(teacher, "doctor_roi_coverage"),
                "multibone_doctor_roi_coverage": as_float(multibone, "doctor_roi_coverage"),
                "teacher_bone_overlap": as_float(teacher, "pred_bone_overlap"),
                "multibone_bone_overlap": as_float(multibone, "pred_bone_overlap"),
                "multibone_score": as_float(multibone, "score"),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    mean_error = sum(float(row["center_error_mm"]) for row in rows) / count
    mean_iou = sum(float(row["pred_box_doctor_bbox_iou"]) for row in rows) / count
    mean_coverage = sum(float(row["doctor_roi_coverage"]) for row in rows) / count
    mean_bone = sum(float(row["pred_bone_overlap"]) for row in rows) / count
    summary = out.with_name("hybrid_teacher_multibone_summary.txt")
    summary.write_text(
        "\n".join(
            [
                "Hybrid teacher + multi-bone traditional locator",
                f"cases: {count}",
                f"score_threshold: {args.score_threshold:g}",
                f"mean_center_error_mm: {mean_error:.2f}",
                f"mean_bbox_iou: {mean_iou:.4f}",
                f"mean_doctor_roi_coverage: {mean_coverage:.4f}",
                f"mean_pred_bone_overlap: {mean_bone:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
