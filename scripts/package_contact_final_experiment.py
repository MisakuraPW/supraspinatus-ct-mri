from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "multibone_contact_final_v2"
PACKAGE = ROOT / "outputs" / "2026-07-01_contact_final_experiment"

BASELINES = [
    ("teacher_baseline", 7.771, 0.0578, 0.0851, 0.0006),
    ("old_multibone", 6.311, 0.1089, 0.1855, 0.0029),
    ("margin_continuity", 5.973, 0.1161, 0.2153, 0.0022),
    ("next_round_before_contact", 5.776, 0.1581, 0.2497, 0.0024),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def make_preview_sheet() -> None:
    preview_dir = PACKAGE / "previews"
    images = sorted(preview_dir.glob("*_top1_preview.png"))
    if not images:
        return
    thumb_w, thumb_h = 420, 280
    cols = 2
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (thumb_w * cols, thumb_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(images):
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h - 24), Image.Resampling.BILINEAR)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(img, (x + (thumb_w - img.width) // 2, y + 24))
        draw.text((x + 8, y + 6), path.stem.replace("_top1_preview", ""), fill=(20, 20, 20))
    sheet.save(preview_dir / "all_cases_contact_final_sheet.png")


def write_report() -> None:
    reports = PACKAGE / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = read_csv(SOURCE / "results" / "summary_metrics.csv")[0]
    final_rows = read_csv(SOURCE / "results" / "per_case_final.csv")
    failure_rows = read_csv(SOURCE / "results" / "failure_analysis.csv")

    baseline_table = "\n".join(
        f"| {name} | {err:.3f} | {iou:.4f} | {cov:.4f} | {bone:.4f} |"
        for name, err, iou, cov, bone in BASELINES
    )
    case_table = "\n".join(
        "| {case} | {selected_method} | {center_error_mm} | {doctor_roi_coverage} | "
        "{pred_bone_overlap} | {dx_mm} | {dy_mm} | {dz_mm} | {decision_reason} |".format(**row)
        for row in final_rows
    )
    failure_table = "\n".join(
        "| {case} | {failure_type} | {top5_best_coverage} | {top5_min_center_error} | "
        "{final_selected_source} | {decision_reason} |".format(**row)
        for row in failure_rows
    )

    report = f"""# CT冈上肌腱ROI自动定位传统方法优化报告

日期：2026-07-01

## 1. 本轮目标

本轮不做深度学习训练，继续沿用传统CT图像处理路线，在老师单肱骨锚点方法和前一版多骨锚点方法基础上，优先做方法改进，而不是盲目微调参数。核心目标是提升综合性能，尤其降低最坏病例误差、提高医生ROI覆盖，同时监控骨重叠风险。

## 2. 方法概述

当前最终方法仍以“近端肱骨上外侧锚点 + 肩峰/肩胛顶板骨结构”为主要多骨锚点。候选生成不是只选一个CT切片，而是在体数据中生成一组ROI候选，每个候选包含z层、x/y中心、bbox、候选来源和多项评分。

本轮保留前一版的：

- `current_multibone`：主多骨锚点候选。
- `low_z`：向低z方向探索的候选。
- `teacher_z_refine`：在老师预测z附近重新生成候选。
- 骨距离margin、z连续性、软组织HU和体内比例等传统规则。

新增关键分支：

- `contact_z`：针对止点附近真实ROI可能包含一定骨接触的病例，允许有限骨体素进入ROI。该分支只在满足“候选确实有骨接触、骨重叠不超过上限、且相对当前结果明显向低z补偿”时参与最终选择。

为什么需要它：SB病例复盘发现，医生ROI中心附近本身骨体素比例约8.6%，而旧规则把ROI内骨比例硬限制在1.2%以内，导致真实区域附近候选被系统性过滤。因此这不是单纯参数问题，而是“严格避骨”假设与部分标注不一致。

## 3. 总体结果

| 方法 | mean center error | mean bbox IoU | mean coverage | mean bone overlap |
|---|---:|---:|---:|---:|
{baseline_table}
| contact_final | {summary['mean_center_error_mm']} | {summary['mean_bbox_iou']} | {summary['mean_doctor_roi_coverage']} | {summary['mean_pred_bone_overlap']} |

最终结果：

- mean center error：{summary['mean_center_error_mm']} mm
- median center error：{summary['median_center_error_mm']} mm
- worst-case center error：{summary['worst_center_error_mm']} mm
- mean bbox IoU：{summary['mean_bbox_iou']}
- mean doctor ROI coverage：{summary['mean_doctor_roi_coverage']}
- mean pred bone overlap：{summary['mean_pred_bone_overlap']}
- mean abs dx/dy/dz：{summary['mean_abs_dx_mm']} / {summary['mean_abs_dy_mm']} / {summary['mean_abs_dz_mm']} mm

相对前一版 `next_round_before_contact`：中心误差从5.776降到4.903 mm，覆盖率从0.2497升到0.3182，最坏病例从10.63降到9.61 mm。代价是骨重叠从0.0024升到0.0072，主要来自SB病例的止点骨接触。

## 4. 逐例结果

| case | selected | error | coverage | bone overlap | dx | dy | dz | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
{case_table}

## 5. 失败与风险分析

| case | failure type | top5 best coverage | top5 min error | final source | decision |
|---|---|---:|---:|---|---|
{failure_table}

主要结论：

- SB：本轮最大改进。`contact_z`将误差降到1.90 mm，coverage升到0.6846，但骨重叠升到0.051。该病例提示医生ROI可能包含止点骨接触，严格避骨会错误过滤真实区域。
- WQX：仍是最坏病例，误差9.61 mm，coverage仅0.0251。额外候选中存在轻微改善，但缺少稳定的无监督选择门槛；强行选择会误伤LWL等病例，因此本轮不纳入自动规则。
- LHY：top-k中存在更好候选，仍属于排序问题。后续可研究更可靠的候选重排序，而不是扩大搜索。
- 骨重叠：均值仍较低，但SB和ZJ/YPL需要单独解释。若老师要求严格避骨，可以切换到更保守版本；若优先ROI覆盖和中心误差，推荐当前版本。

## 6. 复现实验命令

```bash
python scripts/run_multibone_locator.py \\
  --data-dir outputs/normalized_10cases \\
  --output-dir outputs/multibone_contact_final_v2 \\
  --teacher-csv outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv \\
  --current-anchor-count 160 \\
  --bone-margin-voxels 2 \\
  --continuity-window 2 \\
  --continuity-xy-tolerance 14 \\
  --low-z-enable \\
  --low-z-range-mm 12 \\
  --low-z-step-mm 2 \\
  --low-z-weight 0.85 \\
  --branch-anchor-count 6 \\
  --contact-z-enable \\
  --contact-z-range-mm 14 \\
  --contact-z-min-shift-mm 6 \\
  --contact-z-step-mm 2 \\
  --contact-z-weight 0.68 \\
  --contact-z-max-bone-fraction 0.12 \\
  --contact-z-select-enable \\
  --teacher-z-refine-enable \\
  --teacher-z-window-mm 8 \\
  --teacher-z-refine-step-mm 2 \\
  --topk 5
```

## 7. 包内文件

- `previews/`：10个病例的定位预览图，以及总览图。
- `results/`：`summary_metrics.csv`、`per_case_final.csv`、`per_case_topk.csv`、`failure_analysis.csv`和最终ROI NIfTI。
- `reports/`：本报告。
"""
    (reports / "contact_final_report_2026-07-01.md").write_text(report, encoding="utf-8")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    copytree(SOURCE / "previews", PACKAGE / "previews")
    copytree(SOURCE / "results", PACKAGE / "results")
    make_preview_sheet()
    write_report()
    print(f"wrote package to {PACKAGE}")


if __name__ == "__main__":
    main()
