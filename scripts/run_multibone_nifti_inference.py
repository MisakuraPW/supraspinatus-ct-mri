from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import NiftiImage, load_nifti, save_nifti_like
from supraspinatus_locator.localization.multi_bone_traditional import (
    MultiBoneConfig,
    candidate_quality_flags,
    locate_multibone_candidates,
    make_external_guided_threshold_mask,
    prediction_mask_and_bbox,
    save_candidate_sheet_pil,
    save_overlay_montage_pil,
    select_adaptive_edge_guarded,
    select_consensus_candidate,
    select_scored_candidate,
    write_csv_union,
)
from supraspinatus_locator.preprocessing.totalseg_bones import load_mask_compatible


def find_ct_60kev(case_dir: Path) -> Path:
    ct_dir = case_dir / "CT"
    matches = sorted(path for path in ct_dir.iterdir() if path.is_file() and "60" in path.name.lower() and ".nii" in path.name.lower())
    if not matches:
        raise FileNotFoundError(f"No 60keV NIfTI found under {ct_dir}")
    return matches[0]


def discover_cases(data_dir: Path, cases: set[str] | None = None) -> list[Path]:
    out = []
    for path in sorted((p for p in data_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        if cases is not None and path.name not in cases:
            continue
        if (path / "CT").exists():
            out.append(path)
    return out


def candidate_row(case_name: str, idx: int, candidate, image: np.ndarray, spacing: np.ndarray, cfg: MultiBoneConfig) -> tuple[dict[str, object], np.ndarray]:
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


def source_top_rows(rows: list[dict[str, object]], per_source_topk: int) -> list[dict[str, object]]:
    source_order = ["current_multibone", "surface_arc", "bone_edge_tendon", "low_z", "contact_z", "teacher_z_refine", "teacher_low_z", "wide_xy"]
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


def select_candidate(rows: list[dict[str, object]], policy: str) -> tuple[dict[str, object], str]:
    sorted_rows = sorted(rows, key=lambda row: float(row["total_score"]), reverse=True)
    if policy == "best_score":
        return sorted_rows[0], "choose_highest_raw_total_score"
    if policy == "current_first":
        current = [row for row in sorted_rows if row.get("candidate_source") == "current_multibone"]
        return (current[0], "choose_current_multibone_first") if current else (sorted_rows[0], "fallback_highest_raw_total_score")
    if policy == "consensus":
        return select_consensus_candidate(rows)
    if policy == "conservative":
        return select_scored_candidate(rows, mode="conservative")
    if policy == "edge_priority":
        return select_scored_candidate(rows, mode="edge_priority")
    if policy == "surface_suppressed":
        return select_scored_candidate(rows, mode="surface_suppressed")
    if policy == "adaptive_edge_guarded":
        return select_adaptive_edge_guarded(rows)
    return sorted_rows[0], "choose_highest_raw_total_score"


def load_external_bone_mask(
    image: np.ndarray,
    cfg: MultiBoneConfig,
    case_name: str,
    bone_mask_dir: Path | None,
    bone_mask_filename: str,
    allow_threshold_bone_fallback: bool,
    min_external_bone_voxels: int,
    external_bone_mode: str,
    external_bone_dilation_voxels: int,
) -> tuple[np.ndarray | None, str, int | str]:
    if bone_mask_dir is None:
        return None, "threshold", ""
    external_bone_path = bone_mask_dir / case_name / bone_mask_filename
    if not external_bone_path.exists():
        if allow_threshold_bone_fallback:
            return None, f"threshold_fallback_missing_external:{external_bone_path}", ""
        raise FileNotFoundError(f"External bone mask not found for {case_name}: {external_bone_path}")
    external_mask = load_mask_compatible(external_bone_path, image.shape)
    external_voxels = int(np.asarray(external_mask, dtype=bool).sum())
    if external_voxels < min_external_bone_voxels:
        if allow_threshold_bone_fallback:
            return None, f"threshold_fallback_invalid_external:{external_bone_path}", external_voxels
        raise ValueError(f"External bone mask too small for {case_name}: {external_voxels}")
    if external_bone_mode == "direct":
        return external_mask, str(external_bone_path), external_voxels
    guided = make_external_guided_threshold_mask(image, external_mask, cfg.bone_threshold_hu, external_bone_dilation_voxels)
    guided_voxels = int(np.asarray(guided, dtype=bool).sum())
    if guided_voxels < min_external_bone_voxels:
        if allow_threshold_bone_fallback:
            return None, f"threshold_fallback_sparse_guided_external:{external_bone_path}", external_voxels
        raise ValueError(f"Guided bone mask too small for {case_name}: {guided_voxels}")
    return guided, f"threshold_roi:{external_bone_path}:dilation={external_bone_dilation_voxels}:voxels={guided_voxels}", external_voxels


def run_inference(
    data_dir: Path,
    output_dir: Path,
    cfg: MultiBoneConfig,
    args: argparse.Namespace,
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
    cases = set(args.cases) if args.cases else None

    for case_dir in discover_cases(data_dir, cases):
        try:
            image_obj: NiftiImage = load_nifti(find_ct_60kev(case_dir))
            image = image_obj.data.astype(np.float32)
            spacing = np.asarray(image_obj.spacing[:3], dtype=float)
            cfg.spacing_xyz = tuple(float(v) for v in spacing)
            cfg.teacher_z_center = None
            cfg.teacher_center_xyz = None
            bone_mask_override, bone_mask_source, external_voxels = load_external_bone_mask(
                image,
                cfg,
                case_dir.name,
                Path(args.bone_mask_dir) if args.bone_mask_dir else None,
                args.bone_mask_filename,
                args.allow_threshold_bone_fallback,
                args.min_external_bone_voxels,
                args.external_bone_mode,
                args.external_bone_dilation_voxels,
            )
            candidates = locate_multibone_candidates(image, cfg, top_k=max(cfg.top_k * 4, 20), bone_mask_override=bone_mask_override)
        except Exception as exc:
            failures.append({"case": case_dir.name, "error": str(exc)})
            continue

        masks: dict[str, np.ndarray] = {}
        rows: list[dict[str, object]] = []
        for idx, candidate in enumerate(candidates, start=1):
            row, mask = candidate_row(case_dir.name, idx, candidate, image, spacing, cfg)
            row["bone_mask_source"] = bone_mask_source
            row["external_bone_voxels"] = external_voxels
            rows.append(row)
            masks[str(row["candidate_id"])] = mask

        candidate_pool_rows = source_top_rows(rows, cfg.top_k)
        selected, decision_reason = select_candidate(candidate_pool_rows, args.selection_policy)
        selected_id = str(selected["candidate_id"])
        selected_mask = masks[selected_id]
        final_row = {
            **selected,
            "method": "nifti_unlabeled_inference",
            "selection_policy": args.selection_policy,
            "selected_candidate_id": selected_id,
            "selected_method": selected["candidate_source"],
            "decision_reason": decision_reason,
            "ct_60_file": str(find_ct_60kev(case_dir)),
        }
        final_rows.append(final_row)
        topk_rows.extend(candidate_pool_rows)

        preview_rows = candidate_pool_rows[: max(1, int(args.candidate_preview_topk))]
        if args.export_candidates:
            preview_path = previews_dir / f"{case_dir.name}_candidate_top{len(preview_rows)}.png"
            save_candidate_sheet_pil(image, preview_path, preview_rows, masks, doctor_roi=None)
            for row in preview_rows[:5]:
                feedback_rows.append({"case": case_dir.name, "candidate_id": row["candidate_id"], "visual_label": "", "note": "", "preview_path": str(preview_path)})

        save_nifti_like(results_dir / f"{case_dir.name}_final_roi.nii.gz", selected_mask.astype(np.uint8), reference=image_obj)
        save_overlay_montage_pil(image, previews_dir / f"{case_dir.name}_preview.png", masks=[(selected_mask, (64, 220, 255))], center=int(round(float(selected["pred_center_z"]))))

    write_csv_union(results_dir / "per_case_inference.csv", final_rows)
    write_csv_union(results_dir / "per_case_topk.csv", topk_rows)
    write_csv_union(results_dir / "failures.csv", failures)
    if feedback_rows:
        write_csv_union(results_dir / "unlabeled_visual_feedback_template.csv", feedback_rows)
    if final_rows:
        bone_overlap = np.asarray([float(row["pred_bone_overlap"]) for row in final_rows])
        source_counts = Counter(str(row["selected_method"]) for row in final_rows)
        write_csv_union(
            results_dir / "summary_inference.csv",
            [
                {
                    "method": "nifti_unlabeled_inference",
                    "cases": len(final_rows),
                    "failures": len(failures),
                    "mean_pred_bone_overlap": round(float(bone_overlap.mean()), 4),
                    "max_pred_bone_overlap": round(float(bone_overlap.max()), 4),
                    "selected_source_counts": "; ".join(f"{k}:{v}" for k, v in sorted(source_counts.items())),
                    "selection_policy": args.selection_policy,
                }
            ],
        )
        report = [
            "# Unlabeled NIfTI inference",
            "",
            f"data_dir: {data_dir}",
            f"bone_mask_dir: {args.bone_mask_dir or 'HU threshold'}",
            f"selection_policy: {args.selection_policy}",
            f"cases_processed: {len(final_rows)}",
            f"failures: {len(failures)}",
            f"mean_pred_bone_overlap: {bone_overlap.mean():.4f}",
            f"selected_source_counts: {dict(source_counts)}",
            "",
            "No doctor ROI labels were available, so this run reports inference outputs and internal safety/contact indicators only.",
        ]
    else:
        report = ["# Unlabeled NIfTI inference", "", "No cases were processed successfully."]
    (reports_dir / "unlabeled_nifti_inference_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-bone locator on unlabeled project-style NIfTI CT cases.")
    parser.add_argument("--data-dir", default="outputs/2026-07_unlabel_ct_nifti")
    parser.add_argument("--output-dir", default="outputs/2026-07_unlabeled_nifti_locator")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--selection-policy", choices=("best_score", "current_first", "conservative", "consensus", "edge_priority", "surface_suppressed", "adaptive_edge_guarded"), default="adaptive_edge_guarded")
    parser.add_argument("--export-candidates", action="store_true")
    parser.add_argument("--candidate-preview-topk", type=int, default=8)
    parser.add_argument("--bone-mask-dir", default=None)
    parser.add_argument("--bone-mask-filename", default="shoulder_bones_fused_hu.nii.gz")
    parser.add_argument("--allow-threshold-bone-fallback", action="store_true")
    parser.add_argument("--min-external-bone-voxels", type=int, default=10000)
    parser.add_argument("--external-bone-mode", choices=("threshold_roi", "direct"), default="threshold_roi")
    parser.add_argument("--external-bone-dilation-voxels", type=int, default=4)
    parser.add_argument("--bone-threshold", type=float, default=300.0)
    parser.add_argument("--half-size", nargs=3, type=int, default=(22, 8, 2))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--current-anchor-count", type=int, default=160)
    parser.add_argument("--surface-arc-enable", action="store_true")
    parser.add_argument("--surface-arc-anchor-count", type=int, default=8)
    parser.add_argument("--bone-edge-enable", action="store_true")
    parser.add_argument("--bone-edge-anchor-count", type=int, default=8)
    parser.add_argument("--bone-edge-channel-downshift-voxels", type=float, default=0.0)
    args = parser.parse_args()

    cfg = MultiBoneConfig(
        bone_threshold_hu=args.bone_threshold,
        half_size=tuple(args.half_size),
        top_k=args.topk,
        current_anchor_count=args.current_anchor_count,
        surface_arc_enable=args.surface_arc_enable,
        surface_arc_anchor_count=args.surface_arc_anchor_count,
        bone_edge_enable=args.bone_edge_enable,
        bone_edge_anchor_count=args.bone_edge_anchor_count,
        bone_edge_channel_downshift_voxels=args.bone_edge_channel_downshift_voxels,
    )
    run_inference(Path(args.data_dir), Path(args.output_dir), cfg, args)
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
