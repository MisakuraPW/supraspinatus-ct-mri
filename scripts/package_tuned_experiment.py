from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "multibone_next_round"
TUNING = ROOT / "outputs" / "multibone_tuned_selection"
PACKAGE = ROOT / "outputs" / "2026-07-01_tuned_experiment"


OLD_RESULTS = [
    ("teacher_baseline", 7.771, 0.0578, 0.0851, 0.0006),
    ("old_multibone", 6.311, 0.1089, 0.1855, 0.0029),
    ("margin_continuity", 5.973, 0.1161, 0.2153, 0.0022),
    ("old_hybrid", 5.954, 0.1178, 0.2075, 0.0020),
    ("previous_best_hybrid", 5.776, 0.1248, 0.2197, 0.0015),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_report() -> None:
    reports = PACKAGE / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = read_csv(SOURCE / "results" / "summary_metrics.csv")[0]
    final_rows = read_csv(SOURCE / "results" / "per_case_final.csv")
    failures = read_csv(SOURCE / "results" / "failure_analysis.csv")
    tuned = read_csv(TUNING / "selection_tuning_best_params.csv")[0] if (TUNING / "selection_tuning_best_params.csv").exists() else None

    old_rows = "\n".join(f"| {name} | {err:.3f} | {iou:.4f} | {cov:.4f} | {bone:.4f} |" for name, err, iou, cov, bone in OLD_RESULTS)
    case_rows = "\n".join(
        f"| {r['case']} | {r['selected_method']} | {r['center_error_mm']} | {r['dx_mm']} | {r['dy_mm']} | {r['dz_mm']} | {r['doctor_roi_coverage']} | {r['pred_bone_overlap']} | {r['decision_reason']} |"
        for r in final_rows
    )
    failure_rows = "\n".join(
        f"| {r['case']} | {r['failure_type']} | {r['top5_best_coverage']} | {r['top5_min_center_error']} | {r['decision_reason']} |"
        for r in failures
    )
    tuned_text = "未运行离线融合规则调参。"
    if tuned is not None:
        tuned_text = (
            f"离线融合规则调参得到 conservative selection：mean error {tuned['mean_center_error_mm']} mm，"
            f"coverage {tuned['mean_doctor_roi_coverage']}，bone overlap {tuned['mean_pred_bone_overlap']}。"
            "它降低了骨重叠，但 coverage 低于推荐结果，因此本包推荐继续使用 next_round final。"
        )
    report = f"""# 传统多骨定位精细调参与综合性能报告

日期：2026-07-01

## 1. 当前推荐方法

本包推荐使用 `next_round_final`：传统多骨候选生成 + low-z exploration + teacher-z-refine 候选 + 骨表面距离带评分 + 多因素融合规则。

这里的“CT 候选生成”不是只选一个切片，而是在 CT 体数据中生成一批 ROI 候选。每个候选包含：

- z 层面；
- x/y 中心；
- ROI bbox；
- 候选来源；
- 解剖评分、软组织评分、骨风险评分、连续性评分。

因此，“找哪个切片”是候选生成里最重要的一部分。SB 当前仍失败，说明低 z 层候选生成还没有覆盖医生 ROI 所在层面。

## 2. 方法详解

老师 baseline 使用单一上外侧肱骨锚点。我们的当前方法在此基础上加入肩峰/肩胛骨顶板结构，形成肱骨-顶板多骨锚点约束。

本轮主要精细化了两类内容：

1. 融合/排序调参：读取已生成 top-k 候选，离线搜索多因素决策规则。这个过程不重新跑 CT 候选生成，速度快。
2. 候选生成调参脚本：将原来的巨大全排列改成少量可解释策略档位，例如扩大 low-z 搜索、放宽/收紧骨距离、调整 z 连续性。

## 3. 总体性能

| 方法 | mean center error | mean bbox IoU | mean coverage | mean bone overlap |
|---|---:|---:|---:|---:|
{old_rows}
| next_round_final | {summary['mean_center_error_mm']} | {summary['mean_bbox_iou']} | {summary['mean_doctor_roi_coverage']} | {summary['mean_pred_bone_overlap']} |

推荐结果：

- mean center error: {summary['mean_center_error_mm']} mm
- median center error: {summary['median_center_error_mm']} mm
- worst-case center error: {summary['worst_center_error_mm']} mm
- mean bbox IoU: {summary['mean_bbox_iou']}
- mean doctor ROI coverage: {summary['mean_doctor_roi_coverage']}
- mean bone overlap: {summary['mean_pred_bone_overlap']}

方向误差：

- mean abs dx/dy/dz: {summary['mean_abs_dx_mm']} / {summary['mean_abs_dy_mm']} / {summary['mean_abs_dz_mm']} mm
- median abs dx/dy/dz: {summary['median_abs_dx_mm']} / {summary['median_abs_dy_mm']} / {summary['median_abs_dz_mm']} mm

## 4. 离线调参结果

{tuned_text}

结论：当前提升空间主要不在融合权重，而在候选生成，特别是 SB/WQX 的 z/xy 候选覆盖。

## 5. 每例结果

| case | selected | error | dx | dy | dz | coverage | bone | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
{case_rows}

## 6. Top-k 与失败类型

| case | failure type | top5 best coverage | top5 min error | decision |
|---|---|---:|---:|---|
{failure_rows}

## 7. 后续候选生成调参方式

推荐先跑少量策略档位：

```bash
python scripts/grid_search_multibone_locator.py \\
  --data-dir outputs/normalized_10cases \\
  --output-dir outputs/multibone_grid_search_0701 \\
  --topk 5
```

这会跑默认 8 个可解释 profile，不会做巨大全排列。如果只想试前 2 个：

```bash
python scripts/grid_search_multibone_locator.py \\
  --data-dir outputs/normalized_10cases \\
  --output-dir outputs/multibone_grid_search_0701 \\
  --topk 5 \\
  --max-runs 2
```

如果后续确认某个 profile 更好，再围绕它做小范围细化。

## 8. 文件说明

- `previews/`：处理预览图、总览图、重点病例 top5 候选图。
- `results/`：最终结果、top-k、failure analysis、summary metrics。
- `tuning/`：离线融合规则调参结果。
- `reports/`：本报告。
"""
    (reports / "tuned_experiment_report_2026-07-01.md").write_text(report, encoding="utf-8")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    copytree(SOURCE / "previews", PACKAGE / "previews")
    copytree(SOURCE / "results", PACKAGE / "results")
    if TUNING.exists():
        copytree(TUNING, PACKAGE / "tuning")
    write_report()
    print(f"wrote package to {PACKAGE}")


if __name__ == "__main__":
    main()
