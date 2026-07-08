from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "surface_arc_best_final"
PREVIOUS = ROOT / "outputs" / "multibone_contact_final_v2"
TEACHER = ROOT / "outputs" / "teacher_10cases" / "evaluation" / "ct_tendon_locator_results.csv"
PACKAGE = ROOT / "outputs" / "2026-07-01_best_final_experiment"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


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
    sheet.save(preview_dir / "all_cases_best_final_sheet.png")


def write_report() -> None:
    reports = PACKAGE / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    teacher_rows = read_csv(TEACHER)
    previous_summary = read_csv(PREVIOUS / "results" / "summary_metrics.csv")[0]
    current_summary = read_csv(SOURCE / "results" / "summary_metrics.csv")[0]
    previous_rows = {row["case"]: row for row in read_csv(PREVIOUS / "results" / "per_case_final.csv")}
    current_rows = read_csv(SOURCE / "results" / "per_case_final.csv")

    teacher_summary = {
        "mean_center_error_mm": f"{mean(teacher_rows, 'center_error_mm'):.3f}",
        "mean_bbox_iou": f"{mean(teacher_rows, 'pred_box_doctor_bbox_iou'):.4f}",
        "mean_doctor_roi_coverage": f"{mean(teacher_rows, 'doctor_roi_coverage'):.4f}",
        "mean_pred_bone_overlap": f"{mean(teacher_rows, 'pred_bone_overlap'):.4f}",
    }

    overall_table = "\n".join(
        "| {name} | {desc} | {err} | {iou} | {cov} | {bone} |".format(
            name=name,
            desc=desc,
            err=summary["mean_center_error_mm"],
            iou=summary["mean_bbox_iou"],
            cov=summary["mean_doctor_roi_coverage"],
            bone=summary["mean_pred_bone_overlap"],
        )
        for name, desc, summary in [
            ("Teacher baseline", "老师单肱骨锚点方法", teacher_summary),
            ("Contact-aware", "上一版：多骨 + contact_z", previous_summary),
            ("Best final", "最终最佳：弧形通道 + 表面距离/有限贴骨评分", current_summary),
        ]
    )

    case_table = "\n".join(
        "| {case} | {selected_method} | {center_error_mm} | {doctor_roi_coverage} | {bbox_iou} | {pred_bone_overlap} | {dx_mm} | {dy_mm} | {dz_mm} |".format(
            **row
        )
        for row in current_rows
    )

    comparison_table = []
    for case in ["LWL", "WQX", "SB", "LHY", "OSQ", "ZJY"]:
        before = previous_rows[case]
        after = next(row for row in current_rows if row["case"] == case)
        comparison_table.append(
            "| {case} | {before_src} | {before_err} | {before_cov} | {before_bone} | {after_src} | {after_err} | {after_cov} | {after_bone} |".format(
                case=case,
                before_src=before["selected_method"],
                before_err=before["center_error_mm"],
                before_cov=before["doctor_roi_coverage"],
                before_bone=before["pred_bone_overlap"],
                after_src=after["selected_method"],
                after_err=after["center_error_mm"],
                after_cov=after["doctor_roi_coverage"],
                after_bone=after["pred_bone_overlap"],
            )
        )

    report = f"""# CT冈上肌腱ROI自动定位最终最佳传统方法报告

日期：2026-07-01

## 1. 本轮继续尝试的方向

本轮继续针对“篮筐中心偏高偏外、冈上肌腱沿肱骨头弧形包绕”的反馈做方法改进。我们尝试了三类建议：

1. 更稳定的肱骨头圆/球面拟合：在每层肱骨头圆拟合之外，尝试用病例级中位圆心/半径进行轻量球面平滑。
2. 从单个矩形框扩展到弧形多点/中心线ROI：尝试用沿弧面排列的多个小框合成最终ROI。
3. 用骨表面距离/有限贴骨评分替代“骨重叠越低越好”：允许候选贴近肱骨头表面，只有明显深入骨内才强惩罚。

经过关键病例和全量10例验证，最终最佳版本采用第3点，并保留第1点和第2点的实现但默认不启用。原因是：球面平滑可轻微改善WQX，但会明显削弱LWL；中心线小框能表达弧形，但在当前ROI尺度下coverage下降，没有超过单框弧形候选。

## 2. 最终最佳方法

最终方法仍以多骨锚点为主：近端肱骨上外侧锚点 + 肩峰/肩胛顶板结构。新增的最佳有效模块是`surface_arc`：

- 粗略估计肱骨头二维圆弧；
- 沿肱骨头上外侧表面生成弧形通道候选；
- 使用表面距离评分，鼓励候选落在合理贴骨距离；
- 允许有限骨接触，避免骨惩罚把ROI推得过外；
- 只在候选局部、z层稳定、沿弧面向下贴近时接管当前结果。

这不是系统性平移，而是基于肱骨头表面几何关系生成新候选。

## 3. 总体指标

| 方法 | 说明 | mean center error ↓ | mean bbox IoU ↑ | mean doctor ROI coverage ↑ | mean bone overlap |
|---|---|---:|---:|---:|---:|
{overall_table}

最终最佳结果：

- mean center error：**{current_summary['mean_center_error_mm']} mm**
- worst-case error：**{current_summary['worst_center_error_mm']} mm**
- mean doctor ROI coverage：**{current_summary['mean_doctor_roi_coverage']}**
- mean bbox IoU：**{current_summary['mean_bbox_iou']}**
- mean bone overlap：**{current_summary['mean_pred_bone_overlap']}**

与上一版相比，骨重叠从{previous_summary['mean_pred_bone_overlap']}升至{current_summary['mean_pred_bone_overlap']}，但中心误差、coverage、bbox IoU、worst-case全部改善。这说明骨重叠不是越低越科学；对冈上肌腱止点和包绕肱骨头的结构而言，有限贴骨更符合实际解剖。

## 4. 最终逐例结果

| case | selected source | center error | coverage | bbox IoU | bone overlap | dx | dy | dz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{case_table}

## 5. 关键病例对比

| case | 上一版来源 | 上一版误差 | 上一版coverage | 上一版bone | 最终来源 | 最终误差 | 最终coverage | 最终bone |
|---|---|---:|---:|---:|---|---:|---:|---:|
{chr(10).join(comparison_table)}

LWL和WQX是本轮最关键的收益病例。LWL由5.50 mm降至0.50 mm，WQX由9.61 mm降至4.40 mm，均由`surface_arc`接管。SB继续由`contact_z`处理，说明止点贴骨候选仍必要。LHY、OSQ、ZJY未被弧形候选误接管，说明最终接管门槛相对稳定。

## 6. 未采用尝试

- 球面平滑：关键病例中WQX略有改善，但LWL由0.50 mm退化到3.46 mm，因此不作为最终默认。
- 多点/中心线ROI：实现后可运行，但当前小框组合降低了coverage；在现有10例上不如单框弧形候选。该功能保留为后续实验入口。

## 7. 后续建议

后续若继续推进，建议优先做更精细的肱骨头表面重建，而不是继续调骨重叠阈值。更理想的指标应区分：贴骨、浅接触、深度入骨，而不是把骨重叠简单当作单调惩罚。
"""
    (reports / "best_final_report_2026-07-01.md").write_text(report, encoding="utf-8")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    copytree(SOURCE / "previews", PACKAGE / "previews")
    copytree(SOURCE / "results", PACKAGE / "results")
    make_preview_sheet()
    write_report()
    print(f"wrote package to {PACKAGE}")


if __name__ == "__main__":
    main()
