from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "case",
    "teacher_center_error_mm",
    "multibone_center_error_mm",
    "delta_center_error_mm",
    "teacher_bbox_iou",
    "multibone_bbox_iou",
    "delta_bbox_iou",
    "teacher_doctor_roi_coverage",
    "multibone_doctor_roi_coverage",
    "delta_doctor_roi_coverage",
    "teacher_bone_overlap",
    "multibone_bone_overlap",
]


def read_csv_by_case(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare teacher single-anchor locator and our multi-bone locator.")
    parser.add_argument("--teacher-csv", default="outputs/teacher_lhy/evaluation/ct_tendon_locator_results.csv")
    parser.add_argument("--multibone-csv", default="outputs/multibone_lhy/multibone_locator_results.csv")
    parser.add_argument("--out", default="outputs/multibone_lhy/teacher_vs_multibone.csv")
    args = parser.parse_args()

    teacher = read_csv_by_case(Path(args.teacher_csv))
    multibone = read_csv_by_case(Path(args.multibone_csv))
    rows = []
    for case in sorted(set(teacher) & set(multibone)):
        t = teacher[case]
        m = multibone[case]
        rows.append(
            {
                "case": case,
                "teacher_center_error_mm": f(t, "center_error_mm"),
                "multibone_center_error_mm": f(m, "center_error_mm"),
                "delta_center_error_mm": f(t, "center_error_mm") - f(m, "center_error_mm"),
                "teacher_bbox_iou": f(t, "pred_box_doctor_bbox_iou"),
                "multibone_bbox_iou": f(m, "pred_box_doctor_bbox_iou"),
                "delta_bbox_iou": f(m, "pred_box_doctor_bbox_iou") - f(t, "pred_box_doctor_bbox_iou"),
                "teacher_doctor_roi_coverage": f(t, "doctor_roi_coverage"),
                "multibone_doctor_roi_coverage": f(m, "doctor_roi_coverage"),
                "delta_doctor_roi_coverage": f(m, "doctor_roi_coverage") - f(t, "doctor_roi_coverage"),
                "teacher_bone_overlap": f(t, "pred_bone_overlap"),
                "multibone_bone_overlap": f(m, "pred_bone_overlap"),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

