from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.localization.roi_geometry import BBox3D, mask_from_bbox


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "multibone_next_round"
DATA = ROOT / "outputs" / "normalized_10cases"
TEACHER_CSV = ROOT / "outputs" / "teacher_10cases" / "evaluation" / "ct_tendon_locator_results.csv"
FOCUS_CASES = ("SB", "WQX", "ZJ", "OSQ")
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


def by_case(path: Path) -> dict[str, dict[str, str]]:
    return {row["case"]: row for row in read_csv(path)}


def find_first(directory: Path, token: str) -> Path:
    return next(p for p in directory.iterdir() if p.is_file() and token.lower() in p.name.lower() and ".nii" in p.name.lower())


def bbox_from_row(row: dict[str, str], prefix: str = "pred_box") -> BBox3D:
    return BBox3D(
        (int(float(row[f"{prefix}_x1"])), int(float(row[f"{prefix}_y1"])), int(float(row[f"{prefix}_z1"]))),
        (int(float(row[f"{prefix}_x2"])), int(float(row[f"{prefix}_y2"])), int(float(row[f"{prefix}_z2"]))),
    )


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_2d[np.isfinite(slice_2d)], [1, 99])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((slice_2d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def draw_mask_bbox(panel: Image.Image, mask_slice: np.ndarray, color: tuple[int, int, int]) -> None:
    pts = np.argwhere(mask_slice > 0)
    if len(pts) == 0:
        return
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    sx = panel.size[0] / mask_slice.shape[1]
    sy = panel.size[1] / mask_slice.shape[0]
    ImageDraw.Draw(panel).rectangle((int(x1 * sx), int(y1 * sy), int((x2 + 1) * sx), int((y2 + 1) * sy)), outline=color, width=4)


def panel(image: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]], z: int, title: str, body: str) -> Image.Image:
    z = max(0, min(image.shape[2] - 1, z))
    base = Image.fromarray(normalize_slice(image[:, :, z].T), mode="L").convert("RGBA").resize((320, 320))
    for mask, color in masks:
        draw_mask_bbox(base, mask[:, :, z].T, color)
    canvas = Image.new("RGB", (320, 380), "white")
    canvas.paste(base.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 326), title, fill=(20, 20, 20))
    draw.text((8, 348), body, fill=(60, 60, 60))
    return canvas


def make_previews() -> None:
    preview_dir = OUT / "previews"
    topk_dir = preview_dir / "topk_overlays"
    preview_dir.mkdir(parents=True, exist_ok=True)
    topk_dir.mkdir(parents=True, exist_ok=True)
    teacher = by_case(TEACHER_CSV)
    final = by_case(OUT / "results" / "per_case_final.csv")
    topk_rows = read_csv(OUT / "results" / "per_case_topk.csv")
    contact_panels: list[Image.Image] = []
    for case, row in final.items():
        ct_dir = DATA / case / "CT"
        image = load_nifti(find_first(ct_dir, "60")).data.astype(np.float32)
        doctor = load_nifti(find_first(ct_dir, "roi")).data != 0
        final_mask = mask_from_bbox(image.shape, bbox_from_row(row))
        teacher_mask = mask_from_bbox(image.shape, bbox_from_row(teacher[case]))
        doctor_z = int(round(float(np.argwhere(doctor).mean(axis=0)[2])))
        final_z = int(round(float(row["pred_center_z"])))
        teacher_z = int(round(float(teacher[case]["pred_center_z"])))
        masks = [(doctor, (255, 50, 60)), (teacher_mask, (255, 210, 0)), (final_mask, (0, 220, 255))]
        sheet = Image.new("RGB", (960, 380), "white")
        for idx, img in enumerate(
            [
                panel(image, masks, teacher_z, f"{case} teacher z={teacher_z}", f"err {float(teacher[case]['center_error_mm']):.2f}"),
                panel(image, masks, final_z, f"{case} final z={final_z}", f"err {float(row['center_error_mm']):.2f} cov {float(row['doctor_roi_coverage']):.3f}"),
                panel(image, masks, doctor_z, f"{case} doctor z={doctor_z}", f"src {row['selected_method']}"),
            ]
        ):
            sheet.paste(img, (idx * 320, 0))
        sheet.save(preview_dir / f"{case}_final_contact.png")
        contact_panels.append(sheet.resize((480, 190), Image.Resampling.BILINEAR))

        if case in FOCUS_CASES:
            raw_top = [r for r in topk_rows if r["case"] == case]
            case_top = [row] + [r for r in raw_top if r["candidate_id"] != row["candidate_id"]]
            case_top = case_top[:5]
            top_sheet = Image.new("RGB", (320 * len(case_top), 380), "white")
            for idx, cand in enumerate(case_top):
                cand_mask = mask_from_bbox(image.shape, bbox_from_row(cand))
                z = int(round(float(cand["pred_center_z"])))
                title = f"{case} {cand['candidate_source']} #{cand.get('global_rank', cand.get('rank', idx + 1))}"
                body = f"err {float(cand['center_error_mm']):.2f} cov {float(cand['doctor_roi_coverage']):.3f} bone {float(cand['bone_overlap']):.4f}"
                top_sheet.paste(panel(image, [(doctor, (255, 50, 60)), (cand_mask, (0, 220, 255))], z, title, body), (idx * 320, 0))
            top_sheet.save(topk_dir / f"{case}_top5_candidates.png")

    contact = Image.new("RGB", (960, 950), "white")
    for idx, img in enumerate(contact_panels):
        contact.paste(img, ((idx % 2) * 480, (idx // 2) * 190))
    contact.save(preview_dir / "all_cases_final_contact_sheet.png")


def write_report() -> None:
    reports = OUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = read_csv(OUT / "results" / "summary_metrics.csv")[0]
    final = read_csv(OUT / "results" / "per_case_final.csv")
    failures = read_csv(OUT / "results" / "failure_analysis.csv")
    old_rows = "\n".join(f"| {name} | {err:.3f} | {iou:.4f} | {cov:.4f} | {bone:.4f} |" for name, err, iou, cov, bone in OLD_RESULTS)
    case_rows = "\n".join(
        f"| {r['case']} | {r['selected_method']} | {r['center_error_mm']} | {r['dx_mm']} | {r['dy_mm']} | {r['dz_mm']} | {r['doctor_roi_coverage']} | {r['pred_bone_overlap']} | {r['decision_reason']} |"
        for r in final
    )
    failure_rows = "\n".join(
        f"| {r['case']} | {r['failure_type']} | {r['top5_best_coverage']} | {r['top5_min_center_error']} | {r['decision_reason']} |"
        for r in failures
    )
    report = f"""# 下一轮传统多骨定位实验报告

日期：2026-06-29

## 方法变化

本轮没有引入深度学习，仍沿用传统 CT 图像处理框架。主要新增：误差方向分解、low-z exploration、teacher-z-refine 候选、骨表面距离带评分、多来源 top-k 候选、多因素融合规则。

## 总体结果

| 方法 | mean center error | mean bbox IoU | mean coverage | mean bone overlap |
|---|---:|---:|---:|---:|
{old_rows}
| next_round_final | {summary['mean_center_error_mm']} | {summary['mean_bbox_iou']} | {summary['mean_doctor_roi_coverage']} | {summary['mean_pred_bone_overlap']} |

本轮 mean center error 为 {summary['mean_center_error_mm']} mm，coverage 为 {summary['mean_doctor_roi_coverage']}，worst-case center error 为 {summary['worst_center_error_mm']} mm。

## 方向误差

- mean abs dx/dy/dz: {summary['mean_abs_dx_mm']} / {summary['mean_abs_dy_mm']} / {summary['mean_abs_dz_mm']} mm
- median abs dx/dy/dz: {summary['median_abs_dx_mm']} / {summary['median_abs_dy_mm']} / {summary['median_abs_dz_mm']} mm

## 每例最终结果

| case | selected | error | dx | dy | dz | coverage | bone | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
{case_rows}

## Top-k 与失败类型

| case | failure type | top5 best coverage | top5 min error | decision |
|---|---|---:|---:|---|
{failure_rows}

## 重点病例结论

- SB：仍为 generation_failure，top5 coverage 为 0，说明 low-z 分支仍未把正确层面纳入有效候选。
- WQX：仍为 generation_failure/弱覆盖，主要为 y 方向偏移。
- ZJ：被识别为 possible bone margin shift，最终回退 teacher baseline，避免过强避骨造成劣化。
- OSQ：coverage 明显提高，但仍存在 xy 偏移风险，需要后续针对 x/y 方向约束。

## 回答验收问题

- 是否降低 mean center error：与老师和旧多骨相比降低；与上一版最好融合持平。
- 是否提高 coverage：相比所有旧结果均提高。
- 是否降低 worst-case error：相比旧多骨和老师降低，但 SB 仍是 worst case。
- 改善来自候选生成还是排序规则：ZJ 主要来自融合排序规则；OSQ 主要来自当前多骨候选排序；SB 仍是候选生成失败。
- 是否存在骨重叠风险上升：mean bone overlap 高于上一版最好融合，但低于旧多骨，OSQ/YPL/ZJ 需继续单独监控。
"""
    (reports / "experiment_report_next_round.md").write_text(report, encoding="utf-8")


def main() -> None:
    make_previews()
    write_report()
    print(f"wrote next-round package under {OUT}")


if __name__ == "__main__":
    main()
