from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.localization.roi_geometry import BBox3D, mask_from_bbox


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "outputs" / "6.24"
DATA_DIR = ROOT / "outputs" / "normalized_10cases"
TEACHER_CSV = ROOT / "outputs" / "teacher_10cases" / "evaluation" / "ct_tendon_locator_results.csv"
MULTIBONE_CSV = ROOT / "outputs" / "multibone_10cases_margin_continuity" / "multibone_locator_results.csv"
HYBRID_CSV = ROOT / "outputs" / "hybrid_10cases_margin_continuity" / "hybrid_teacher_multibone_results.csv"
COMPARE_CSV = ROOT / "outputs" / "multibone_10cases_margin_continuity" / "teacher_vs_multibone_margin_continuity.csv"
TOPK_CSV = ROOT / "outputs" / "multibone_10cases_margin_continuity" / "multibone_topk_candidates.csv"


def read_csv_by_case(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def find_first(directory: Path, token: str) -> Path:
    matches = [p for p in directory.iterdir() if p.is_file() and token.lower() in p.name.lower() and ".nii" in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No file containing {token!r} under {directory}")
    return matches[0]


def bbox_from_row(row: dict[str, str], prefix: str) -> BBox3D:
    return BBox3D(
        (
            int(float(row[f"{prefix}_x1"])),
            int(float(row[f"{prefix}_y1"])),
            int(float(row[f"{prefix}_z1"])),
        ),
        (
            int(float(row[f"{prefix}_x2"])),
            int(float(row[f"{prefix}_y2"])),
            int(float(row[f"{prefix}_z2"])),
        ),
    )


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    finite = slice_2d[np.isfinite(slice_2d)]
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((slice_2d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def overlay_mask(base: Image.Image, mask_slice: np.ndarray, color: tuple[int, int, int], alpha: int = 120) -> None:
    if not np.any(mask_slice):
        return
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask_slice > 0).astype(np.uint8) * alpha, mode="L")
    color_img = Image.new("RGBA", base.size, (*color, alpha))
    overlay.paste(color_img, (0, 0), mask_img)
    base.alpha_composite(overlay)


def draw_slice_bbox(base: Image.Image, mask_slice: np.ndarray, color: tuple[int, int, int], width: int = 4) -> None:
    points = np.argwhere(mask_slice > 0)
    if len(points) == 0:
        return
    y1, x1 = points.min(axis=0)
    y2, x2 = points.max(axis=0)
    scale_x = base.size[0] / mask_slice.shape[1]
    scale_y = base.size[1] / mask_slice.shape[0]
    rect = (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int((x2 + 1) * scale_x),
        int((y2 + 1) * scale_y),
    )
    ImageDraw.Draw(base).rectangle(rect, outline=color, width=width)


def make_panel(
    image: np.ndarray,
    doctor_mask: np.ndarray,
    teacher_mask: np.ndarray,
    multibone_mask: np.ndarray,
    z: int,
    title: str,
    body: str,
) -> Image.Image:
    z = max(0, min(image.shape[2] - 1, z))
    gray = normalize_slice(image[:, :, z].T)
    panel = Image.fromarray(gray, mode="L").convert("RGBA")
    teacher_slice = teacher_mask[:, :, z].T
    multibone_slice = multibone_mask[:, :, z].T
    doctor_slice = doctor_mask[:, :, z].T
    overlay_mask(panel, teacher_slice, (255, 215, 0), 120)
    overlay_mask(panel, multibone_slice, (0, 220, 255), 120)
    overlay_mask(panel, doctor_slice, (255, 55, 65), 145)
    panel = panel.resize((360, 360), Image.Resampling.BILINEAR)
    draw_slice_bbox(panel, teacher_slice, (255, 215, 0), width=4)
    draw_slice_bbox(panel, multibone_slice, (0, 220, 255), width=4)
    draw_slice_bbox(panel, doctor_slice, (255, 55, 65), width=4)
    canvas = Image.new("RGB", (360, 430), "white")
    canvas.paste(panel.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 360, 359, 429), fill=(248, 248, 248))
    draw.text((10, 368), title, fill=(20, 20, 20))
    draw.text((10, 390), body, fill=(60, 60, 60))
    draw.text((10, 412), "red=doctor  yellow=teacher  cyan=ours", fill=(80, 80, 80))
    return canvas


def case_preview(
    case: str,
    teacher_row: dict[str, str],
    multibone_row: dict[str, str],
    hybrid_row: dict[str, str],
    out_dir: Path,
) -> Image.Image:
    ct_dir = DATA_DIR / case / "CT"
    image = load_nifti(find_first(ct_dir, "60")).data.astype(np.float32)
    doctor_mask = load_nifti(find_first(ct_dir, "roi")).data != 0
    teacher_mask = mask_from_bbox(image.shape, bbox_from_row(teacher_row, "pred_box")).astype(bool)
    multibone_mask = mask_from_bbox(image.shape, bbox_from_row(multibone_row, "pred_box")).astype(bool)

    teacher_z = int(round(float(teacher_row["pred_center_z"])))
    multibone_z = int(round(float(multibone_row["pred_center_z"])))
    doctor_z = int(round(float(np.argwhere(doctor_mask).mean(axis=0)[2])))
    panels = [
        make_panel(
            image,
            doctor_mask,
            teacher_mask,
            multibone_mask,
            teacher_z,
            f"{case} teacher z={teacher_z}",
            f"err {float(teacher_row['center_error_mm']):.2f}mm cov {float(teacher_row['doctor_roi_coverage']):.3f}",
        ),
        make_panel(
            image,
            doctor_mask,
            teacher_mask,
            multibone_mask,
            multibone_z,
            f"{case} ours z={multibone_z}",
            f"err {float(multibone_row['center_error_mm']):.2f}mm cov {float(multibone_row['doctor_roi_coverage']):.3f}",
        ),
        make_panel(
            image,
            doctor_mask,
            teacher_mask,
            multibone_mask,
            doctor_z,
            f"{case} doctor z={doctor_z}",
            f"hybrid {hybrid_row['selected_method'].split('_')[0]} bone {float(hybrid_row['pred_bone_overlap']):.4f}",
        ),
    ]
    canvas = Image.new("RGB", (1080, 430), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * 360, 0))
    canvas.save(out_dir / f"{case}_comparison_preview.png")
    return canvas


def write_summary_csv(package_results_dir: Path, teacher_rows: list[dict[str, str]], multibone_rows: list[dict[str, str]], hybrid_rows: list[dict[str, str]]) -> None:
    rows = [
        {
            "method": "teacher_single_humerus_anchor",
            "mean_center_error_mm": f"{mean(teacher_rows, 'center_error_mm'):.3f}",
            "mean_bbox_iou": f"{mean(teacher_rows, 'pred_box_doctor_bbox_iou'):.4f}",
            "mean_doctor_roi_coverage": f"{mean(teacher_rows, 'doctor_roi_coverage'):.4f}",
            "mean_pred_bone_overlap": f"{mean(teacher_rows, 'pred_bone_overlap'):.4f}",
        },
        {
            "method": "ours_multibone_margin_continuity",
            "mean_center_error_mm": f"{mean(multibone_rows, 'center_error_mm'):.3f}",
            "mean_bbox_iou": f"{mean(multibone_rows, 'pred_box_doctor_bbox_iou'):.4f}",
            "mean_doctor_roi_coverage": f"{mean(multibone_rows, 'doctor_roi_coverage'):.4f}",
            "mean_pred_bone_overlap": f"{mean(multibone_rows, 'pred_bone_overlap'):.4f}",
        },
        {
            "method": "hybrid_teacher_fallback",
            "mean_center_error_mm": f"{mean(hybrid_rows, 'center_error_mm'):.3f}",
            "mean_bbox_iou": f"{mean(hybrid_rows, 'pred_box_doctor_bbox_iou'):.4f}",
            "mean_doctor_roi_coverage": f"{mean(hybrid_rows, 'doctor_roi_coverage'):.4f}",
            "mean_pred_bone_overlap": f"{mean(hybrid_rows, 'pred_bone_overlap'):.4f}",
        },
    ]
    out = package_results_dir / "method_summary_metrics.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(package_dir: Path, teacher_rows: list[dict[str, str]], multibone_rows: list[dict[str, str]], hybrid_rows: list[dict[str, str]]) -> None:
    teacher = {
        "err": mean(teacher_rows, "center_error_mm"),
        "iou": mean(teacher_rows, "pred_box_doctor_bbox_iou"),
        "cov": mean(teacher_rows, "doctor_roi_coverage"),
        "bone": mean(teacher_rows, "pred_bone_overlap"),
    }
    ours = {
        "err": mean(multibone_rows, "center_error_mm"),
        "iou": mean(multibone_rows, "pred_box_doctor_bbox_iou"),
        "cov": mean(multibone_rows, "doctor_roi_coverage"),
        "bone": mean(multibone_rows, "pred_bone_overlap"),
    }
    hybrid = {
        "err": mean(hybrid_rows, "center_error_mm"),
        "iou": mean(hybrid_rows, "pred_box_doctor_bbox_iou"),
        "cov": mean(hybrid_rows, "doctor_roi_coverage"),
        "bone": mean(hybrid_rows, "pred_bone_overlap"),
    }
    rows_by_case = "\n".join(
        "| {case} | {selected_method} | {center_error_mm} | {doctor_roi_coverage} | {pred_bone_overlap} |".format(**row)
        for row in hybrid_rows
    )
    report = f"""# CT 中冈上肌腱候选 ROI 自动定位实验报告

日期：2026-06-24

## 1. 实验目标

本阶段目标不是完成冈上肌腱精细分割，而是在肩关节 CT 中自动定位冈上肌腱止点附近的候选采样 ROI。该 ROI 后续可作为 CT-MRI 多模态分割、胶原蛋白预测或医生复核的前置定位结果。

当前使用 10 个病例进行初步验证：HMC、LHY、LWL、OSQ、SB、WQX、YPL、ZH、ZJ、ZJY。

## 2. 方法概述

老师基线方法使用单一上外侧肱骨锚点。流程为：60keV CT 输入、骨阈值分割、寻找近端肱骨组件、根据肱骨上外侧锚点生成候选 ROI，并用软组织强度、骨重叠等规则打分。

本次改进方法仍为传统图像处理，不使用深度学习。主要增加三点：

1. 多骨锚点：在肱骨上外侧锚点基础上，引入肩峰/肩胛骨顶板结构，用“肱骨-顶板”关系约束冈上肌腱止点附近走廊。
2. 骨距离 margin：除惩罚 ROI 内部骨体素外，进一步惩罚 ROI 外扩 1-3 voxel 范围内贴骨的候选，降低“贴骨但未明显重叠”的风险。
3. z 层连续性：候选不只由单层骨组件决定，要求相邻 z 层存在相近的肱骨-顶板结构，提高层面稳定性。

此外，保留老师方法作为低置信保底。当多骨候选置信分低于阈值时，融合策略回退到老师单肱骨锚点结果。

## 3. 评价指标

- center error mm：预测 ROI 中心与医生 ROI 中心的三维距离，越低越好。
- bbox IoU：预测 ROI bbox 与医生 ROI bbox 的交并比，越高越好。
- doctor ROI coverage：预测 ROI 覆盖医生 ROI 的比例，越高越好。
- bone overlap：预测 ROI 中骨体素比例，越低越好。

注意：当前医生 ROI 更接近“采样/参考 ROI”，不是完整肌腱精细分割标签。

## 4. 总体结果

| 方法 | mean center error mm | mean bbox IoU | mean doctor ROI coverage | mean bone overlap |
|---|---:|---:|---:|---:|
| 老师基线：单肱骨锚点 | {teacher['err']:.3f} | {teacher['iou']:.4f} | {teacher['cov']:.4f} | {teacher['bone']:.4f} |
| 本方法：多骨 + margin + z 连续性 | {ours['err']:.3f} | {ours['iou']:.4f} | {ours['cov']:.4f} | {ours['bone']:.4f} |
| 融合：老师保底 + 本方法 | {hybrid['err']:.3f} | {hybrid['iou']:.4f} | {hybrid['cov']:.4f} | {hybrid['bone']:.4f} |

与老师基线相比，融合方案的平均中心误差由 {teacher['err']:.3f} mm 降至 {hybrid['err']:.3f} mm，doctor ROI coverage 由 {teacher['cov']:.4f} 提升至 {hybrid['cov']:.4f}。骨重叠仍保持在较低水平，平均为 {hybrid['bone']:.4f}。

## 5. 每例融合结果

| 病例 | 选择方法 | center error mm | doctor ROI coverage | bone overlap |
|---|---|---:|---:|---:|
{rows_by_case}

## 6. 现象与讨论

多骨锚点整体提升了定位精度和覆盖率，说明单一肱骨上外侧锚点只能描述“止点附近在哪里”，而加入肩峰/肩胛骨顶板后，可以进一步描述“冈上肌腱从哪里来、往哪里走”。

骨距离 margin 有助于降低贴骨风险。典型例子是 YPL，骨重叠从旧多骨版本的 0.0102 降至 0.0063。z 层连续性则改善了部分层面偏移问题，例如 HMC 和 OSQ。

当前仍存在失败样本。SB 病例中，候选主要停留在较高 z 层，未覆盖医生 ROI 所在层面；这说明后续需要增加低 z 分支，或在老师 z 层附近增加多骨微调候选。

## 7. 结论

当前传统图像处理方案在 10 例初步数据上优于老师基线，尤其在中心误差、bbox IoU 和医生 ROI 覆盖率上均有提升。下一步建议继续围绕 SB 类 z 层失败样本优化候选生成，并将融合规则从单一 score 阈值扩展为 score、margin、continuity、teacher/multibone z 差异共同决策。

## 8. 文件说明

- `previews/`：10 个病例的三联对比预览图，以及总 contact sheet。
- `results/`：老师基线、本方法、融合方法的 CSV 指标表。
- `experiment_report_2026-06-24.md`：本报告。
"""
    (package_dir / "experiment_report_2026-06-24.md").write_text(report, encoding="utf-8")


def main() -> None:
    previews_dir = PACKAGE_DIR / "previews"
    results_dir = PACKAGE_DIR / "results"
    previews_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    teacher_by_case = read_csv_by_case(TEACHER_CSV)
    multibone_by_case = read_csv_by_case(MULTIBONE_CSV)
    hybrid_by_case = read_csv_by_case(HYBRID_CSV)
    teacher_rows = read_rows(TEACHER_CSV)
    multibone_rows = read_rows(MULTIBONE_CSV)
    hybrid_rows = read_rows(HYBRID_CSV)

    preview_images = []
    for case in sorted(hybrid_by_case):
        preview_images.append(case_preview(case, teacher_by_case[case], multibone_by_case[case], hybrid_by_case[case], previews_dir))

    thumb_w, thumb_h = 540, 215
    contact = Image.new("RGB", (thumb_w * 2, thumb_h * 5), "white")
    for idx, image in enumerate(preview_images):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        x = (idx % 2) * thumb_w
        y = (idx // 2) * thumb_h
        contact.paste(thumb, (x, y))
    contact.save(previews_dir / "all_cases_comparison_contact_sheet.png")

    shutil.copy2(TEACHER_CSV, results_dir / "teacher_ct_tendon_locator_results.csv")
    shutil.copy2(MULTIBONE_CSV, results_dir / "ours_multibone_margin_continuity_results.csv")
    shutil.copy2(HYBRID_CSV, results_dir / "hybrid_teacher_fallback_results.csv")
    shutil.copy2(COMPARE_CSV, results_dir / "teacher_vs_ours_multibone_margin_continuity.csv")
    shutil.copy2(TOPK_CSV, results_dir / "ours_multibone_topk_candidates.csv")
    write_summary_csv(results_dir, teacher_rows, multibone_rows, hybrid_rows)
    write_report(PACKAGE_DIR, teacher_rows, multibone_rows, hybrid_rows)
    print(f"wrote package to {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
