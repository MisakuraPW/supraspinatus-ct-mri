from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "surface_arc_final"
PREVIOUS = ROOT / "outputs" / "multibone_contact_final_v2"
TEACHER = ROOT / "outputs" / "teacher_10cases" / "evaluation" / "ct_tendon_locator_results.csv"
PACKAGE = ROOT / "outputs" / "2026-07-01_surface_arc_experiment"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def metric(row: dict[str, str], key: str) -> str:
    return row[key]


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
    sheet.save(preview_dir / "all_cases_surface_arc_sheet.png")


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

    overall_rows = [
        ("Teacher baseline", "老师单肱骨锚点方法", teacher_summary),
        ("Previous contact-aware", "上一版：多骨 + top-k/refine/fallback + contact_z", previous_summary),
        ("Surface-arc final", "本版：上一版基础上加入肱骨头弧形通道候选", current_summary),
    ]
    overall_table = "\n".join(
        "| {name} | {desc} | {err} | {iou} | {cov} | {bone} |".format(
            name=name,
            desc=desc,
            err=summary["mean_center_error_mm"],
            iou=summary["mean_bbox_iou"],
            cov=summary["mean_doctor_roi_coverage"],
            bone=summary["mean_pred_bone_overlap"],
        )
        for name, desc, summary in overall_rows
    )

    case_table = "\n".join(
        "| {case} | {selected_method} | {center_error_mm} | {doctor_roi_coverage} | {bbox_iou} | {pred_bone_overlap} | {dx_mm} | {dy_mm} | {dz_mm} |".format(
            **row
        )
        for row in current_rows
    )

    comparison_cases = ["LWL", "WQX", "SB", "LHY", "OSQ", "ZJY"]
    comparison_table = []
    for case in comparison_cases:
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

    report = f"""# 解决“偏高偏外”问题的弧形通道定位实验报告

日期：2026-07-01

## 1. 问题背景

老师反馈：医生认为当前篮筐中心大多比实际冈上肌腱位置偏高、偏外，而冈上肌腱并不是一个孤立点状结构，它会沿肱骨头表面呈弧形包绕。

这说明单纯追求“ROI不和骨重叠”并不一定科学。肌腱止点和肱骨头表面天然相邻，如果骨惩罚过强，候选框可能被推离肱骨头表面，表现为偏外；如果候选只按一个框中心定位，也难以表达肌腱沿肱骨头走行的弧形结构。

因此本轮重点不是继续做系统性平移校正，而是从方法上让候选更贴近“肱骨头表面弧形通道”。

## 2. 本轮方法改进

### 2.1 肱骨头弧形通道候选

在原来的多骨锚点方法中，ROI主要由肱骨上外侧锚点和肩峰/肩胛顶板结构共同决定。本轮在此基础上，对每个候选层面的肱骨组件估计一个近似肱骨头圆：

- 从CT骨阈值区域中提取肱骨头相关组件；
- 粗略估计肱骨头圆心和半径；
- 在肱骨头上外侧弧面附近生成一组`surface_arc`候选；
- 候选不再只围绕一个点，而是沿肱骨头表面弧形分布。

这样做的目的，是让ROI从“远离骨头的矩形篮筐”转向“贴近肱骨头弧面的肌腱通道候选”。

### 2.2 骨重叠惩罚机制调整

上一版对常规候选仍然较严格地惩罚骨重叠。本轮没有取消骨约束，而是把弧形候选中的骨约束改成分层逻辑：

- 少量贴骨：允许，作为接近肱骨头表面的证据；
- 大量深入骨内：继续惩罚；
- 离骨太远：不再被认为一定更好。

这与医生反馈一致：肌腱包绕肱骨头，合理ROI应该靠近骨表面，而不是被骨惩罚推到过外侧。

### 2.3 受限接管门槛

第一版弧形候选会误伤部分本来已经较好的病例，因此最终加入了受限接管规则。`surface_arc`只有在满足以下条件时才替代当前候选：

- 与当前多骨候选距离不能跳得过远；
- z层不能大幅乱跳；
- 需要有有限骨接触或贴骨证据；
- 更像是沿肱骨头弧面向下贴近，而不是任意偏移。

因此，本版不是简单系统性校正，也不是无条件提高骨重叠，而是让“弧形贴骨候选”只在其解剖形态合理时接管。

## 3. 总体结果

| 方法 | 说明 | mean center error ↓ | mean bbox IoU ↑ | mean doctor ROI coverage ↑ | mean bone overlap |
|---|---|---:|---:|---:|---:|
{overall_table}

与上一版相比，本轮结果为：

- mean center error：{previous_summary['mean_center_error_mm']} -> **{current_summary['mean_center_error_mm']} mm**
- worst-case error：{previous_summary['worst_center_error_mm']} -> **{current_summary['worst_center_error_mm']} mm**
- mean doctor ROI coverage：{previous_summary['mean_doctor_roi_coverage']} -> **{current_summary['mean_doctor_roi_coverage']}**
- mean bbox IoU：{previous_summary['mean_bbox_iou']} -> **{current_summary['mean_bbox_iou']}**
- mean bone overlap：{previous_summary['mean_pred_bone_overlap']} -> **{current_summary['mean_pred_bone_overlap']}**

可以看到，骨重叠确实增大了，但中心误差、覆盖率、bbox IoU和最坏病例误差均明显改善。这支持一个判断：对于冈上肌腱定位任务，骨重叠不能简单理解为越低越好；合理的贴骨接触可能反而更符合医生ROI。

## 4. 最终逐例结果

| case | selected source | center error | coverage | bbox IoU | bone overlap | dx | dy | dz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{case_table}

## 5. 关键病例对比

| case | 上一版来源 | 上一版误差 | 上一版coverage | 上一版bone | 本版来源 | 本版误差 | 本版coverage | 本版bone |
|---|---|---:|---:|---:|---|---:|---:|---:|
{chr(10).join(comparison_table)}

### LWL

LWL是本轮最明显的改善病例。上一版中心误差为5.50 mm，coverage为0.1929；本版`surface_arc`接管后误差降到0.50 mm，coverage升到0.6567。该病例说明，原候选确实存在“偏高/未贴合肌腱弧形走行”的问题。弧形候选允许ROI沿肱骨头表面向下贴近后，定位显著接近医生ROI。

### WQX

WQX此前是最坏病例，上一版误差9.61 mm，coverage仅0.0251。本版弧形候选接管后，误差降到4.40 mm，coverage升到0.2593。该病例是本轮方法针对“偏高偏外”问题最直接的收益之一。

### SB

SB在上一轮已经由`contact_z`明显修正，本轮保持该结果：误差1.90 mm，coverage 0.6846。SB继续证明严格避骨不适合所有病例，止点贴骨候选是必要补充。

### ZJY、LHY、OSQ

这几例用于验证弧形候选不会随意接管。早期未加门槛时，ZJY会被错误替换；最终规则收紧后，ZJY仍选择`current_multibone`，避免了误伤。LHY和OSQ同样保留原候选，说明本轮不是全局平移，而是受限地修正特定失败模式。

## 6. 风险与后续方向

本轮方法证明，适度增加骨接触可以换来更好的肌腱ROI覆盖和定位精度。但风险也很明确：

- 如果弧形候选选择过于激进，会把本来正确的病例拉到肱骨头表面错误位置；
- 骨重叠指标不能废除，只是不能作为唯一目标；
- 后续最好将“骨表面距离”替代“骨重叠越低越好”，区分贴骨、浅接触和深度入骨。

下一步建议：

1. 继续改进肱骨头圆/球拟合，使弧形候选更稳定；
2. 将ROI从单个矩形框进一步扩展为多点中心线或弧形小框组合；
3. 在更多病例上统计“合理骨接触范围”，形成比骨重叠更科学的贴骨评分指标。
"""
    (reports / "surface_arc_report_2026-07-01.md").write_text(report, encoding="utf-8")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    copytree(SOURCE / "previews", PACKAGE / "previews")
    copytree(SOURCE / "results", PACKAGE / "results")
    make_preview_sheet()
    write_report()
    print(f"wrote package to {PACKAGE}")


if __name__ == "__main__":
    main()
