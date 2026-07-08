from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import NiftiImage, save_nifti_like
from supraspinatus_locator.localization.multi_bone_traditional import (
    MultiBoneConfig,
    candidate_quality_flags,
    locate_multibone_candidates,
    prediction_mask_and_bbox,
    save_candidate_sheet_pil,
    save_overlay_montage_pil,
    write_csv_union,
)


@dataclass
class DicomVolume:
    image: NiftiImage
    series_dir: Path
    patient_name: str
    series_description: str


def _natural_key(path: Path) -> tuple[Any, ...]:
    text = path.name
    parts: list[Any] = []
    cur = ""
    for char in text:
        if char.isdigit():
            cur += char
        else:
            if cur:
                parts.append(int(cur))
                cur = ""
            parts.append(char)
    if cur:
        parts.append(int(cur))
    return tuple(parts)


def discover_case_dirs(data_dir: Path, cases: set[str] | None = None) -> list[Path]:
    out = []
    for path in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        if cases is not None and path.name not in cases:
            continue
        if any(path.glob("PA*/ST*/SE*")):
            out.append(path)
    return out


def choose_60kev_series(case_dir: Path) -> Path:
    candidates = []
    for series_dir in sorted(case_dir.glob("PA*/ST*/SE*"), key=lambda p: str(p)):
        files = sorted((p for p in series_dir.iterdir() if p.is_file()), key=_natural_key)
        if not files:
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
        except Exception:
            continue
        desc = str(getattr(ds, "SeriesDescription", "")).lower()
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        score = 0
        if "60" in desc and "kev" in desc:
            score += 100
        if "monochromatic" in desc:
            score += 20
        if abs(slope - 1.0) < 1e-3 and intercept <= -500:
            score += 10
        score += min(len(files), 80) / 100.0
        candidates.append((score, series_dir))
    if not candidates:
        raise FileNotFoundError(f"No readable DICOM series found under {case_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_dicom_series(series_dir: Path) -> DicomVolume:
    slices = []
    for path in sorted((p for p in series_dir.iterdir() if p.is_file()), key=_natural_key):
        ds = pydicom.dcmread(str(path), force=True)
        if not hasattr(ds, "PixelData"):
            continue
        pos = getattr(ds, "ImagePositionPatient", None)
        z_pos = float(pos[2]) if pos is not None and len(pos) >= 3 else None
        inst = int(getattr(ds, "InstanceNumber", len(slices)))
        slices.append((z_pos, inst, path, ds))
    if not slices:
        raise ValueError(f"No pixel data found in {series_dir}")

    if all(item[0] is not None for item in slices):
        slices.sort(key=lambda item: float(item[0]))
    else:
        slices.sort(key=lambda item: item[1])

    arrays = []
    z_positions = []
    for z_pos, _inst, _path, ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arrays.append(arr * slope + intercept)
        if z_pos is not None:
            z_positions.append(float(z_pos))

    # DICOM pixel arrays are row, column. The locator code expects x, y, z.
    volume = np.stack(arrays, axis=0).transpose(2, 1, 0)
    first = slices[0][3]
    pixel_spacing = [float(v) for v in getattr(first, "PixelSpacing", [1.0, 1.0])]
    if len(z_positions) >= 2:
        z_spacing = float(np.median(np.abs(np.diff(sorted(z_positions)))))
        if z_spacing <= 0:
            z_spacing = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    else:
        z_spacing = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), z_spacing)
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing[:3]
    image = NiftiImage(data=volume, spacing=spacing, affine=affine, header=bytes(348), path=None)
    return DicomVolume(
        image=image,
        series_dir=series_dir,
        patient_name=str(getattr(first, "PatientName", "")),
        series_description=str(getattr(first, "SeriesDescription", "")),
    )


def candidate_row(
    case_name: str,
    idx: int,
    candidate,
    image: np.ndarray,
    spacing: np.ndarray,
    cfg: MultiBoneConfig,
) -> tuple[dict[str, object], np.ndarray]:
    (
        candidate_bone_fraction,
        _candidate_mean_value,
        _candidate_below,
        _candidate_above,
        candidate_body_fraction,
        candidate_near_bone_fraction,
        candidate_margin_bone_fraction,
        candidate_bone_distance_mm,
        candidate_band_score,
    ) = candidate.roi_stats
    candidate_mask, candidate_box = prediction_mask_and_bbox(image.shape, candidate, cfg.half_size)
    bone_mask = image > cfg.bone_threshold_hu
    center = np.asarray(candidate.center_xyz, dtype=float)
    row = {
        "case": case_name,
        "candidate_id": f"{case_name}_{idx:03d}",
        "candidate_source": candidate.candidate_source,
        "rank": idx,
        "total_score": round(float(candidate.score), 4),
        "anatomical_score": round(float(candidate.anatomical_score), 4),
        "soft_tissue_score": round(float(candidate.soft_tissue_score), 4),
        "safety_score": round(float(candidate.safety_score), 4),
        "continuity_score": round(float(candidate.continuity_score), 4),
        "same_anchor_support_count": int(candidate.same_anchor_support_count),
        "near_bone_fraction": round(float(candidate_near_bone_fraction), 4),
        "margin_bone_fraction": round(float(candidate_margin_bone_fraction), 4),
        "bone_distance_mm": round(float(candidate_bone_distance_mm), 3),
        "bone_distance_band_score": round(float(candidate_band_score), 4),
        "pred_bone_overlap": round(float((candidate_mask & bone_mask).sum() / max(1, candidate_mask.sum())), 4),
        "bone_overlap": round(float((candidate_mask & bone_mask).sum() / max(1, candidate_mask.sum())), 4),
        "body_inside_fraction": round(float(candidate_body_fraction), 4),
        "teacher_distance_mm": "" if candidate.teacher_distance_mm is None else round(float(candidate.teacher_distance_mm), 3),
        "arc_angle_deg": "" if candidate.arc_angle_deg is None else round(float(candidate.arc_angle_deg), 3),
        "surface_offset_voxels": "" if candidate.surface_offset_voxels is None else round(float(candidate.surface_offset_voxels), 3),
        "radius_normalized_distance": "" if candidate.radius_normalized_distance is None else round(float(candidate.radius_normalized_distance), 4),
        "edge_angle_deg": "" if candidate.edge_angle_deg is None else round(float(candidate.edge_angle_deg), 3),
        "edge_point_x": "" if candidate.edge_point_xyz is None else round(float(candidate.edge_point_xyz[0]), 3),
        "edge_point_y": "" if candidate.edge_point_xyz is None else round(float(candidate.edge_point_xyz[1]), 3),
        "edge_point_z": "" if candidate.edge_point_xyz is None else round(float(candidate.edge_point_xyz[2]), 3),
        "surface_normal_x": "" if candidate.surface_normal_xy is None else round(float(candidate.surface_normal_xy[0]), 5),
        "surface_normal_y": "" if candidate.surface_normal_xy is None else round(float(candidate.surface_normal_xy[1]), 5),
        "surface_distance_mm": "" if candidate.surface_distance_mm is None else round(float(candidate.surface_distance_mm), 3),
        "cortical_edge_support": "" if candidate.cortical_edge_support is None else round(float(candidate.cortical_edge_support), 4),
        "soft_tissue_band_mean": "" if candidate.soft_tissue_band_mean is None else round(float(candidate.soft_tissue_band_mean), 3),
        "soft_tissue_band_std": "" if candidate.soft_tissue_band_std is None else round(float(candidate.soft_tissue_band_std), 3),
        "bone_edge_continuity_score": "" if candidate.bone_edge_continuity_score is None else round(float(candidate.bone_edge_continuity_score), 4),
        "arc_fit_residual": "" if candidate.arc_fit_residual is None else round(float(candidate.arc_fit_residual), 5),
        "bone_edge_tendon_score": "" if candidate.bone_edge_tendon_score is None else round(float(candidate.bone_edge_tendon_score), 4),
        "arc_fit_failed": bool(candidate.arc_fit_failed),
        "pred_center_x": round(float(center[0]), 2),
        "pred_center_y": round(float(center[1]), 2),
        "pred_center_z": round(float(center[2]), 2),
        "center_y_minus_humerus_top": round(float(center[1] - candidate.humerus_anchor.y1), 3),
        "humerus_anchor_cx": round(float(candidate.humerus_anchor.cx), 3),
        "humerus_anchor_cy": round(float(candidate.humerus_anchor.cy), 3),
        "pred_center_phys_x": round(float(center[0] * spacing[0]), 3),
        "pred_center_phys_y": round(float(center[1] * spacing[1]), 3),
        "pred_center_phys_z": round(float(center[2] * spacing[2]), 3),
        "pred_box_x1": candidate_box.min[0],
        "pred_box_y1": candidate_box.min[1],
        "pred_box_z1": candidate_box.min[2],
        "pred_box_x2": candidate_box.max[0],
        "pred_box_y2": candidate_box.max[1],
        "pred_box_z2": candidate_box.max[2],
        "bbox_size_phys_x": round(float((candidate_box.max[0] - candidate_box.min[0] + 1) * spacing[0]), 3),
        "bbox_size_phys_y": round(float((candidate_box.max[1] - candidate_box.min[1] + 1) * spacing[1]), 3),
        "bbox_size_phys_z": round(float((candidate_box.max[2] - candidate_box.min[2] + 1) * spacing[2]), 3),
        "pred_soft_mean_60kev": round(float(image[candidate_mask > 0].mean()), 2),
        "bone_fraction": round(float(candidate_bone_fraction), 4),
        "humerus_anchor_x1": candidate.humerus_anchor.x1,
        "humerus_anchor_y1": candidate.humerus_anchor.y1,
        "humerus_anchor_x2": candidate.humerus_anchor.x2,
        "humerus_anchor_y2": candidate.humerus_anchor.y2,
        "roof_anchor_x1": candidate.roof_anchor.x1,
        "roof_anchor_y1": candidate.roof_anchor.y1,
        "roof_anchor_x2": candidate.roof_anchor.x2,
        "roof_anchor_y2": candidate.roof_anchor.y2,
    }
    row.update(candidate_quality_flags(row))
    return row, candidate_mask


def same_humerus_anchor(row: dict[str, object], reference: dict[str, object]) -> bool:
    dx = abs(float(row["humerus_anchor_cx"]) - float(reference["humerus_anchor_cx"]))
    dy = abs(float(row["humerus_anchor_cy"]) - float(reference["humerus_anchor_cy"]))
    ref_width = max(1.0, float(reference["humerus_anchor_x2"]) - float(reference["humerus_anchor_x1"]) + 1.0)
    ref_height = max(1.0, float(reference["humerus_anchor_y2"]) - float(reference["humerus_anchor_y1"]) + 1.0)
    return dx <= max(55.0, ref_width * 0.75) and dy <= max(32.0, ref_height * 0.60)


def select_candidate(rows: list[dict[str, object]], cfg: MultiBoneConfig, selection_policy: str = "legacy") -> tuple[dict[str, object], str]:
    model_rows = sorted(rows, key=lambda row: float(row["total_score"]), reverse=True)
    current_rows = [row for row in model_rows if row["candidate_source"] == "current_multibone"]
    top1 = current_rows[0] if current_rows else model_rows[0]
    if selection_policy == "best_score":
        return model_rows[0], "choose_best_total_score"
    if selection_policy == "generalized":
        primary = top1
        primary_gap = float(primary["center_y_minus_humerus_top"])
        primary_anchor_width = float(primary["humerus_anchor_x2"]) - float(primary["humerus_anchor_x1"]) + 1.0
        same_anchor_rows = [row for row in model_rows if same_humerus_anchor(row, primary)]
        surface_rows = [
            row
            for row in same_anchor_rows
            if row["candidate_source"] == "surface_arc"
            and -2.0 <= float(row["center_y_minus_humerus_top"]) <= 32.0
            and 0.010 <= float(row["bone_overlap"]) <= min(0.085, cfg.surface_arc_max_bone_fraction)
            and float(row.get("near_bone_fraction") or 0.0) <= 0.11
            and float(row.get("margin_bone_fraction") or 0.0) <= 0.13
            and float(row.get("body_inside_fraction") or 0.0) >= 0.94
            and float(row["total_score"]) >= float(primary["total_score"]) - 0.80
        ]
        surface_rows.sort(
            key=lambda row: (
                float(row["total_score"])
                + 0.20 * min(1.0, float(row["bone_overlap"]) / 0.04)
                - 0.02 * abs(float(row["center_y_minus_humerus_top"]) - 8.0)
            ),
            reverse=True,
        )
        if surface_rows and (primary_gap < -6.0 or float(surface_rows[0]["total_score"]) >= float(primary["total_score"]) - 0.20):
            return surface_rows[0], "choose_generalized_same_anchor_surface_arc"
        cross_anchor_surface_rows = [
            row
            for row in model_rows
            if row["candidate_source"] == "surface_arc"
            and -2.0 <= float(row["center_y_minus_humerus_top"]) <= 24.0
            and 0.010 <= float(row["bone_overlap"]) <= min(0.085, cfg.surface_arc_max_bone_fraction)
            and float(row.get("near_bone_fraction") or 0.0) <= 0.105
            and float(row.get("margin_bone_fraction") or 0.0) <= 0.115
            and float(row.get("body_inside_fraction") or 0.0) >= 0.94
            and float(row["total_score"]) >= float(primary["total_score"]) - 0.45
            and (float(row["humerus_anchor_x2"]) - float(row["humerus_anchor_x1"]) + 1.0) >= 55.0
        ]
        cross_anchor_surface_rows.sort(
            key=lambda row: (
                float(row["total_score"])
                + 0.25 * min(1.0, float(row["bone_overlap"]) / 0.04)
                - 0.02 * abs(float(row["center_y_minus_humerus_top"]) - 7.5)
            ),
            reverse=True,
        )
        detached_primary = (
            float(primary["bone_overlap"]) <= 0.001
            and float(primary.get("near_bone_fraction") or 0.0) <= 0.005
            and float(primary.get("margin_bone_fraction") or 0.0) <= 0.005
            and primary_gap <= -2.0
        )
        if detached_primary and primary_anchor_width <= 55.0 and cross_anchor_surface_rows:
            return cross_anchor_surface_rows[0], "choose_generalized_cross_anchor_surface_rescue"
        plausible_rows = [
            row
            for row in same_anchor_rows
            if -6.0 <= float(row["center_y_minus_humerus_top"]) <= 32.0
            and float(row.get("body_inside_fraction") or 0.0) >= 0.94
            and float(row["bone_overlap"]) <= min(0.085, cfg.surface_arc_max_bone_fraction)
        ]
        if plausible_rows and primary_gap < -8.0 and float(plausible_rows[0]["total_score"]) >= float(primary["total_score"]) - 0.50:
            return plausible_rows[0], "choose_generalized_height_plausible_candidate"
        return primary, "choose_generalized_current_anchor"

    selected = top1
    decision_reason = "choose_multibone_high_confidence"
    safe_rows = [
        row
        for row in model_rows
        if float(row["bone_overlap"]) <= 0.006 and float(row.get("near_bone_fraction") or 0.0) <= 0.08
    ]
    if float(top1["bone_overlap"]) > 0.011 or float(top1.get("near_bone_fraction") or 0.0) > 0.12:
        if safe_rows and float(safe_rows[0]["total_score"]) >= float(top1["total_score"]) - 0.45:
            selected = safe_rows[0]
            decision_reason = "choose_topk_lower_bone_risk"

    low_z_rows = [row for row in model_rows if row["candidate_source"] == "low_z"]
    if (
        low_z_rows
        and float(low_z_rows[0]["total_score"]) >= float(top1["total_score"]) - 0.20
        and float(low_z_rows[0]["bone_overlap"]) <= 0.006
        and float(top1.get("continuity_score") or 0.0) < 0.50
    ):
        selected = low_z_rows[0]
        decision_reason = "choose_low_z_candidate_generation_fix"

    surface_arc_rows = [row for row in model_rows if row["candidate_source"] == "surface_arc"]
    top1_x = float(top1["pred_center_x"])
    top1_y = float(top1["pred_center_y"])
    top1_z = float(top1["pred_center_z"])
    qualified_surface_arc_rows = [
        row
        for row in surface_arc_rows
        if float(row["total_score"]) >= float(top1["total_score"]) - 1.20
        and 0.015 <= float(row["bone_overlap"]) <= cfg.surface_arc_max_bone_fraction
        and float(row.get("body_inside_fraction") or 0.0) >= 0.94
        and abs(float(row["pred_center_x"]) - top1_x) <= 18.5
        and 12.0 <= float(row["pred_center_y"]) - top1_y <= 19.5
        and abs(float(row["pred_center_z"]) - top1_z) <= 4.0
        and (row.get("teacher_distance_mm") in ("", None) or float(row.get("teacher_distance_mm") or 0.0) <= 12.0)
    ]
    qualified_surface_arc_rows.sort(
        key=lambda row: (
            float(row["total_score"])
            - 0.04 * float(row.get("teacher_distance_mm") or 0.0)
            - 0.03 * abs(float(row["pred_center_z"]) - top1_z)
            - 0.015 * abs(float(row["pred_center_x"]) - top1_x)
        ),
        reverse=True,
    )
    if cfg.surface_arc_select_enable and qualified_surface_arc_rows:
        selected = qualified_surface_arc_rows[0]
        decision_reason = "choose_surface_arc_humeral_head_candidate"

    return selected, decision_reason


def source_top_rows(rows: list[dict[str, object]], per_source_topk: int) -> list[dict[str, object]]:
    source_order = [
        "current_multibone",
        "surface_arc",
        "bone_edge_tendon",
        "low_z",
        "contact_z",
        "teacher_z_refine",
        "teacher_low_z",
        "wide_xy",
    ]
    selected: list[dict[str, object]] = []
    for source in source_order:
        source_rows = [row for row in rows if row["candidate_source"] == source]
        source_rows.sort(key=lambda row: float(row["total_score"]), reverse=True)
        for rank, row in enumerate(source_rows[:per_source_topk], start=1):
            row["source_rank"] = rank
            selected.append(row)
    selected.sort(key=lambda row: float(row["total_score"]), reverse=True)
    for rank, row in enumerate(selected, start=1):
        row["global_rank"] = rank
    return selected


def run_inference(
    data_dir: Path,
    output_dir: Path,
    cfg: MultiBoneConfig,
    cases: set[str] | None = None,
    selection_policy: str = "legacy",
    export_candidates: bool = False,
    candidate_preview_topk: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    previews_dir = output_dir / "previews"
    reports_dir = output_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    final_rows: list[dict[str, object]] = []
    topk_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    feedback_rows: list[dict[str, object]] = []

    for case_dir in discover_case_dirs(data_dir, cases):
        series_dir = choose_60kev_series(case_dir)
        volume = load_dicom_series(series_dir)
        image = volume.image.data.astype(np.float32)
        spacing = np.asarray(volume.image.spacing[:3], dtype=float)
        cfg.spacing_xyz = tuple(float(v) for v in spacing)
        cfg.teacher_z_center = None
        cfg.teacher_center_xyz = None

        try:
            candidates = locate_multibone_candidates(image, cfg, top_k=max(cfg.top_k * 4, 20))
        except Exception as exc:
            failures.append({"case": case_dir.name, "error": str(exc), "series_dir": str(series_dir)})
            continue

        masks: dict[str, np.ndarray] = {}
        rows: list[dict[str, object]] = []
        for idx, candidate in enumerate(candidates, start=1):
            row, mask = candidate_row(case_dir.name, idx, candidate, image, spacing, cfg)
            row["series_dir"] = str(series_dir)
            row["patient_name"] = volume.patient_name
            row["series_description"] = volume.series_description
            row["spacing_x"] = round(float(spacing[0]), 6)
            row["spacing_y"] = round(float(spacing[1]), 6)
            row["spacing_z"] = round(float(spacing[2]), 6)
            rows.append(row)
            masks[str(row["candidate_id"])] = mask

        candidate_pool_rows = source_top_rows(rows, cfg.top_k)
        selected, decision_reason = select_candidate(candidate_pool_rows, cfg, selection_policy=selection_policy)
        selected_id = str(selected["candidate_id"])
        selected_mask = masks[selected_id]
        final_row = {
            **selected,
            "method": "surface_arc_best_unlabeled_inference",
            "selection_policy": selection_policy,
            "selected_candidate_id": selected_id,
            "selected_method": selected["candidate_source"],
            "decision_reason": decision_reason,
        }
        final_rows.append(final_row)
        topk_rows.extend(candidate_pool_rows)

        if export_candidates:
            preview_rows = candidate_pool_rows[: max(1, int(candidate_preview_topk))]
            preview_path = previews_dir / f"{case_dir.name}_candidate_top{len(preview_rows)}.png"
            save_candidate_sheet_pil(image, preview_path, preview_rows, masks, doctor_roi=None)
            for row in preview_rows[:5]:
                feedback_rows.append(
                    {
                        "case": case_dir.name,
                        "candidate_id": row["candidate_id"],
                        "visual_label": "",
                        "note": "",
                        "preview_path": str(preview_path),
                    }
                )

        save_nifti_like(results_dir / f"{case_dir.name}_final_roi.nii.gz", selected_mask.astype(np.uint8), spacing=volume.image.spacing)
        save_nifti_like(output_dir / f"{case_dir.name}_multibone_roi.nii.gz", selected_mask.astype(np.uint8), spacing=volume.image.spacing)
        save_overlay_montage_pil(
            image,
            previews_dir / f"{case_dir.name}_preview.png",
            masks=[(selected_mask, (64, 220, 255))],
            center=int(round(float(selected["pred_center_z"]))),
        )
        save_overlay_montage_pil(
            image,
            output_dir / f"{case_dir.name}_multibone_preview.png",
            masks=[(selected_mask, (64, 220, 255))],
            center=int(round(float(selected["pred_center_z"]))),
        )

    write_csv_union(results_dir / "per_case_inference.csv", final_rows)
    write_csv_union(results_dir / "per_case_topk.csv", topk_rows)
    write_csv_union(results_dir / "failures.csv", failures)
    if feedback_rows:
        write_csv_union(results_dir / "unlabeled_visual_feedback_template.csv", feedback_rows)

    if final_rows:
        bone_overlap = np.asarray([float(row["pred_bone_overlap"]) for row in final_rows])
        near_bone = np.asarray([float(row["near_bone_fraction"]) for row in final_rows])
        margin_bone = np.asarray([float(row["margin_bone_fraction"]) for row in final_rows])
        source_counts = Counter(str(row["selected_method"]) for row in final_rows)
        summary_rows = [
            {
                "method": "surface_arc_best_unlabeled_inference",
                "cases": len(final_rows),
                "failures": len(failures),
                "mean_pred_bone_overlap": round(float(bone_overlap.mean()), 4),
                "max_pred_bone_overlap": round(float(bone_overlap.max()), 4),
                "mean_near_bone_fraction": round(float(near_bone.mean()), 4),
                "mean_margin_bone_fraction": round(float(margin_bone.mean()), 4),
                "selected_source_counts": "; ".join(f"{k}:{v}" for k, v in sorted(source_counts.items())),
                "selection_policy": selection_policy,
            }
        ]
        write_csv_union(results_dir / "summary_inference.csv", summary_rows)
        text = "\n".join(
            [
                "# Unlabeled 10-case DICOM inference",
                "",
                f"data_dir: {data_dir}",
                f"selection_policy: {selection_policy}",
                f"cases_processed: {len(final_rows)}",
                f"failures: {len(failures)}",
                f"mean_pred_bone_overlap: {bone_overlap.mean():.4f}",
                f"max_pred_bone_overlap: {bone_overlap.max():.4f}",
                f"mean_near_bone_fraction: {near_bone.mean():.4f}",
                f"mean_margin_bone_fraction: {margin_bone.mean():.4f}",
                f"selected_source_counts: {dict(source_counts)}",
                "",
                "No doctor ROI labels were available, so this run reports inference outputs and internal safety/contact indicators only.",
                "Accuracy metrics such as center error, IoU, and ROI coverage require manual supraspinatus tendon annotations.",
            ]
        )
    else:
        text = "# Unlabeled 10-case DICOM inference\n\nNo cases were processed successfully.\n"
    (reports_dir / "unlabeled_inference_summary.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run current best multi-bone locator on unlabeled DICOM CT cases.")
    parser.add_argument("--data-dir", default="Data/unlabel")
    parser.add_argument("--output-dir", default="outputs/unlabeled_dicom_inference")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--selection-policy", choices=("legacy", "best_score", "generalized"), default="legacy")
    parser.add_argument("--export-candidates", action="store_true")
    parser.add_argument("--candidate-preview-topk", type=int, default=8)
    parser.add_argument("--bone-threshold", type=float, default=300.0)
    parser.add_argument("--half-size", nargs=3, type=int, default=(22, 8, 2))
    parser.add_argument("--bone-margin-voxels", type=int, default=3)
    parser.add_argument("--continuity-window", type=int, default=2)
    parser.add_argument("--continuity-xy-tolerance", type=float, default=14.0)
    parser.add_argument("--current-anchor-count", type=int, default=160)
    parser.add_argument("--low-z-enable", action="store_true")
    parser.add_argument("--low-z-range-mm", type=float, default=12.0)
    parser.add_argument("--low-z-step-mm", type=float, default=2.0)
    parser.add_argument("--low-z-weight", type=float, default=0.85)
    parser.add_argument("--branch-anchor-count", type=int, default=6)
    parser.add_argument("--surface-arc-enable", action="store_true")
    parser.add_argument("--surface-arc-select-enable", action="store_true")
    parser.add_argument("--surface-arc-anchor-count", type=int, default=8)
    parser.add_argument("--surface-arc-weight", type=float, default=0.92)
    parser.add_argument("--surface-arc-angle-min-deg", type=float, default=25.0)
    parser.add_argument("--surface-arc-angle-max-deg", type=float, default=82.0)
    parser.add_argument("--surface-arc-angle-step-deg", type=float, default=14.0)
    parser.add_argument("--surface-arc-offset-min-voxels", type=int, default=4)
    parser.add_argument("--surface-arc-offset-max-voxels", type=int, default=18)
    parser.add_argument("--surface-arc-offset-step-voxels", type=int, default=4)
    parser.add_argument("--surface-arc-z-window", type=int, default=1)
    parser.add_argument("--surface-arc-max-bone-fraction", type=float, default=0.12)
    parser.add_argument("--surface-arc-target-bone-fraction", type=float, default=0.035)
    parser.add_argument("--surface-arc-bone-sigma", type=float, default=0.045)
    parser.add_argument("--surface-arc-target-offset-voxels", type=float, default=10.0)
    parser.add_argument("--surface-arc-offset-sigma-voxels", type=float, default=7.0)
    parser.add_argument("--surface-arc-sphere-blend", type=float, default=0.25)
    parser.add_argument("--surface-arc-sphere-anchor-count", type=int, default=24)
    parser.add_argument("--surface-arc-centerline-enable", action="store_true")
    parser.add_argument("--surface-arc-centerline-points", type=int, default=3)
    parser.add_argument("--surface-arc-centerline-angle-step-deg", type=float, default=8.0)
    parser.add_argument("--surface-arc-centerline-half-size", nargs=3, type=int, default=(14, 6, 1))
    parser.add_argument("--bone-edge-enable", action="store_true")
    parser.add_argument("--bone-edge-anchor-count", type=int, default=8)
    parser.add_argument("--bone-edge-weight", type=float, default=0.98)
    parser.add_argument("--bone-edge-angle-min-deg", type=float, default=25.0)
    parser.add_argument("--bone-edge-angle-max-deg", type=float, default=95.0)
    parser.add_argument("--bone-edge-angle-step-deg", type=float, default=10.0)
    parser.add_argument("--bone-edge-offset-min-voxels", type=int, default=4)
    parser.add_argument("--bone-edge-offset-max-voxels", type=int, default=16)
    parser.add_argument("--bone-edge-offset-step-voxels", type=int, default=4)
    parser.add_argument("--bone-edge-z-window", type=int, default=1)
    parser.add_argument("--bone-edge-max-bone-fraction", type=float, default=0.12)
    parser.add_argument("--bone-edge-target-bone-fraction", type=float, default=0.025)
    parser.add_argument("--bone-edge-centerline-enable", action="store_true")
    parser.add_argument("--bone-edge-centerline-points", type=int, default=3)
    parser.add_argument("--bone-edge-centerline-angle-step-deg", type=float, default=8.0)
    parser.add_argument("--bone-edge-centerline-half-size", nargs=3, type=int, default=(14, 6, 1))
    parser.add_argument("--bone-edge-channel-downshift-voxels", type=float, default=0.0)
    parser.add_argument("--bone-dist-min-mm", type=float, default=1.0)
    parser.add_argument("--bone-dist-good-min-mm", type=float, default=3.0)
    parser.add_argument("--bone-dist-good-max-mm", type=float, default=8.0)
    parser.add_argument("--bone-dist-far-mm", type=float, default=15.0)
    parser.add_argument("--bone-dist-band-weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = MultiBoneConfig(
        bone_threshold_hu=args.bone_threshold,
        half_size=tuple(args.half_size),
        bone_margin_voxels=args.bone_margin_voxels,
        continuity_window=args.continuity_window,
        continuity_xy_tolerance=args.continuity_xy_tolerance,
        top_k=args.topk,
        current_anchor_count=args.current_anchor_count,
        low_z_enable=args.low_z_enable,
        low_z_range_mm=args.low_z_range_mm,
        low_z_step_mm=args.low_z_step_mm,
        low_z_weight=args.low_z_weight,
        branch_anchor_count=args.branch_anchor_count,
        surface_arc_enable=args.surface_arc_enable,
        surface_arc_select_enable=args.surface_arc_select_enable,
        surface_arc_anchor_count=args.surface_arc_anchor_count,
        surface_arc_weight=args.surface_arc_weight,
        surface_arc_angle_min_deg=args.surface_arc_angle_min_deg,
        surface_arc_angle_max_deg=args.surface_arc_angle_max_deg,
        surface_arc_angle_step_deg=args.surface_arc_angle_step_deg,
        surface_arc_offset_min_voxels=args.surface_arc_offset_min_voxels,
        surface_arc_offset_max_voxels=args.surface_arc_offset_max_voxels,
        surface_arc_offset_step_voxels=args.surface_arc_offset_step_voxels,
        surface_arc_z_window=args.surface_arc_z_window,
        surface_arc_max_bone_fraction=args.surface_arc_max_bone_fraction,
        surface_arc_target_bone_fraction=args.surface_arc_target_bone_fraction,
        surface_arc_bone_sigma=args.surface_arc_bone_sigma,
        surface_arc_target_offset_voxels=args.surface_arc_target_offset_voxels,
        surface_arc_offset_sigma_voxels=args.surface_arc_offset_sigma_voxels,
        surface_arc_sphere_blend=args.surface_arc_sphere_blend,
        surface_arc_sphere_anchor_count=args.surface_arc_sphere_anchor_count,
        surface_arc_centerline_enable=args.surface_arc_centerline_enable,
        surface_arc_centerline_points=args.surface_arc_centerline_points,
        surface_arc_centerline_angle_step_deg=args.surface_arc_centerline_angle_step_deg,
        surface_arc_centerline_half_size=tuple(args.surface_arc_centerline_half_size),
        bone_edge_enable=args.bone_edge_enable,
        bone_edge_anchor_count=args.bone_edge_anchor_count,
        bone_edge_weight=args.bone_edge_weight,
        bone_edge_angle_min_deg=args.bone_edge_angle_min_deg,
        bone_edge_angle_max_deg=args.bone_edge_angle_max_deg,
        bone_edge_angle_step_deg=args.bone_edge_angle_step_deg,
        bone_edge_offset_min_voxels=args.bone_edge_offset_min_voxels,
        bone_edge_offset_max_voxels=args.bone_edge_offset_max_voxels,
        bone_edge_offset_step_voxels=args.bone_edge_offset_step_voxels,
        bone_edge_z_window=args.bone_edge_z_window,
        bone_edge_max_bone_fraction=args.bone_edge_max_bone_fraction,
        bone_edge_target_bone_fraction=args.bone_edge_target_bone_fraction,
        bone_edge_centerline_enable=args.bone_edge_centerline_enable,
        bone_edge_centerline_points=args.bone_edge_centerline_points,
        bone_edge_centerline_angle_step_deg=args.bone_edge_centerline_angle_step_deg,
        bone_edge_centerline_half_size=tuple(args.bone_edge_centerline_half_size),
        bone_edge_channel_downshift_voxels=args.bone_edge_channel_downshift_voxels,
        bone_dist_min_mm=args.bone_dist_min_mm,
        bone_dist_good_min_mm=args.bone_dist_good_min_mm,
        bone_dist_good_max_mm=args.bone_dist_good_max_mm,
        bone_dist_far_mm=args.bone_dist_far_mm,
        bone_dist_band_weight=args.bone_dist_band_weight,
    )
    run_inference(
        Path(args.data_dir),
        Path(args.output_dir),
        cfg,
        cases=set(args.cases) if args.cases else None,
        selection_policy=args.selection_policy,
        export_candidates=args.export_candidates,
        candidate_preview_topk=args.candidate_preview_topk,
    )
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
