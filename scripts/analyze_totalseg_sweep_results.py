from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bone compare and TotalSeg locator strategy sweep outputs.")
    parser.add_argument("--bone-compare-dir", default="outputs/2026-07_bone_segmentation_compare")
    parser.add_argument("--sweep-dir", default="outputs/2026-07_totalseg_locator_strategy_sweep")
    parser.add_argument("--output-report", default="")
    args = parser.parse_args()

    bone_dir = Path(args.bone_compare_dir)
    sweep_dir = Path(args.sweep_dir)
    bone_rows = read_rows(bone_dir / "bone_segmentation_compare_summary.csv")
    ranked = read_rows(sweep_dir / "ranked_summary_metrics.csv")
    ranker = read_rows(sweep_dir / "combined_ranker_policy_summary.csv")
    per_case = read_rows(sweep_dir / "combined_per_case_final.csv")

    lines: list[str] = ["# TotalSeg 骨后端与定位策略 sweep 分析", ""]
    if bone_rows:
        bad_total = [
            row
            for row in bone_rows
            if f(row, "totalseg_voxels") < 10000 or f(row, "guided_to_morph_fraction") < 0.05
        ]
        lines.extend(
            [
                "## 骨分割质量",
                "",
                md_table(
                    ["case", "HU骨体素", "TotalSeg体素", "guided体素", "guided/HU"],
                    [
                        [
                            row["case"],
                            row["morph_voxels"],
                            row["totalseg_voxels"],
                            row["guided_voxels"],
                            row["guided_to_morph_fraction"],
                        ]
                        for row in bone_rows
                    ],
                ),
                "",
                "异常 TotalSeg 病例：" + (", ".join(row["case"] for row in bad_total) if bad_total else "无明显异常"),
                "",
            ]
        )

    if ranked:
        lines.extend(
            [
                "## 定位策略排名",
                "",
                md_table(
                    ["实验", "策略", "平均误差", "最差误差", "覆盖率", "IoU", "骨重叠"],
                    [
                        [
                            row.get("name", ""),
                            row.get("policy", ""),
                            row.get("mean_center_error_mm", ""),
                            row.get("worst_center_error_mm", ""),
                            row.get("mean_doctor_roi_coverage", ""),
                            row.get("mean_bbox_iou", ""),
                            row.get("mean_pred_bone_overlap", ""),
                        ]
                        for row in ranked
                    ],
                ),
                "",
            ]
        )

    if ranker:
        best_by_policy: dict[str, dict[str, str]] = {}
        for row in ranker:
            if row.get("cases") in ("", "0", None):
                continue
            policy = row.get("policy", "")
            if policy not in best_by_policy or f(row, "mean_center_error_mm", 9999.0) < f(best_by_policy[policy], "mean_center_error_mm", 9999.0):
                best_by_policy[policy] = row
        lines.extend(
            [
                "## 候选级 ranker",
                "",
                md_table(
                    ["ranker策略", "最佳实验", "平均误差", "最差误差", "覆盖率", "IoU"],
                    [
                        [
                            policy,
                            row.get("name", ""),
                            row.get("mean_center_error_mm", ""),
                            row.get("worst_center_error_mm", ""),
                            row.get("mean_doctor_roi_coverage", ""),
                            row.get("mean_bbox_iou", ""),
                        ]
                        for policy, row in sorted(best_by_policy.items())
                    ],
                ),
                "",
                "注意：训练集内 ranker 只代表候选池潜力，泛化优先看 LOOCV。",
                "",
            ]
        )

    if per_case:
        non_oracle = [row for row in per_case if "oracle" not in row.get("experiment", "")]
        cases = sorted({row["case"] for row in non_oracle})
        best_rows = []
        for case in cases:
            rows = [row for row in non_oracle if row["case"] == case]
            if rows:
                best = min(rows, key=lambda row: f(row, "center_error_mm", 9999.0))
                best_rows.append(
                    [
                        case,
                        best.get("experiment", ""),
                        best.get("center_error_mm", ""),
                        best.get("doctor_roi_coverage", ""),
                        best.get("candidate_source", ""),
                        best.get("decision_reason", ""),
                    ]
                )
        lines.extend(
            [
                "## 逐例最佳非 oracle 策略",
                "",
                md_table(["case", "实验", "误差", "覆盖率", "候选源", "选择原因"], best_rows),
                "",
            ]
        )

    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "## 初步结论",
                "",
                f"- 当前最佳真实策略：`{best.get('name', '')}`，平均误差 `{best.get('mean_center_error_mm', '')}` mm。",
                "- 若 oracle 明显优于真实策略，说明候选池仍有潜力，主要瓶颈是选择器。",
                "- 若 TotalSeg 体素为 0 或 guided/HU 很低，应自动回退纯 HU 骨后端。",
                "",
            ]
        )

    output_report = Path(args.output_report) if args.output_report else sweep_dir / "reports" / "sweep_analysis_report.md"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_report}")


if __name__ == "__main__":
    main()
