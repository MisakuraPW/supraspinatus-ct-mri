from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.localization.multi_bone_traditional import MultiBoneConfig, process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traditional multi-bone CT supraspinatus sampling ROI locator.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--output-dir", default="outputs/multibone_locator")
    parser.add_argument("--bone-threshold", type=float, default=300.0)
    parser.add_argument("--half-size", nargs=3, type=int, default=(22, 8, 2))
    parser.add_argument("--bone-margin-voxels", type=int, default=3)
    parser.add_argument("--continuity-window", type=int, default=2)
    parser.add_argument("--continuity-xy-tolerance", type=float, default=14.0)
    parser.add_argument("--teacher-csv", default="outputs/teacher_10cases/evaluation/ct_tendon_locator_results.csv")
    parser.add_argument("--cases", nargs="*", default=None, help="Optional case names to process, e.g. SB WQX.")
    parser.add_argument("--bone-mask-dir", default=None, help="Optional external bone masks: <dir>/<case>/shoulder_bones_combined.nii.gz.")
    parser.add_argument("--bone-mask-filename", default="shoulder_bones_combined.nii.gz")
    parser.add_argument("--allow-threshold-bone-fallback", action="store_true")
    parser.add_argument("--min-external-bone-voxels", type=int, default=10000, help="Treat smaller external bone masks as invalid when fallback is allowed.")
    parser.add_argument("--selection-policy", choices=("legacy", "generalized"), default="legacy")
    parser.add_argument("--export-candidates", action="store_true", help="Export per-case top-k candidate review sheets.")
    parser.add_argument("--candidate-preview-topk", type=int, default=8)
    parser.add_argument("--current-anchor-count", type=int, default=160)
    parser.add_argument("--low-z-enable", action="store_true")
    parser.add_argument("--low-z-range-mm", type=float, default=12.0)
    parser.add_argument("--low-z-step-mm", type=float, default=2.0)
    parser.add_argument("--low-z-weight", type=float, default=0.85)
    parser.add_argument("--branch-anchor-count", type=int, default=6)
    parser.add_argument("--teacher-low-z-enable", action="store_true")
    parser.add_argument("--teacher-low-z-range-mm", type=float, default=16.0)
    parser.add_argument("--teacher-low-z-min-shift-mm", type=float, default=4.0)
    parser.add_argument("--teacher-low-z-step-mm", type=float, default=2.0)
    parser.add_argument("--teacher-low-z-weight", type=float, default=0.88)
    parser.add_argument("--teacher-low-z-select-enable", action="store_true")
    parser.add_argument("--contact-z-enable", action="store_true")
    parser.add_argument("--contact-z-range-mm", type=float, default=14.0)
    parser.add_argument("--contact-z-min-shift-mm", type=float, default=6.0)
    parser.add_argument("--contact-z-step-mm", type=float, default=2.0)
    parser.add_argument("--contact-z-weight", type=float, default=0.68)
    parser.add_argument("--contact-z-max-bone-fraction", type=float, default=0.12)
    parser.add_argument("--contact-z-select-enable", action="store_true")
    parser.add_argument("--teacher-z-refine-enable", action="store_true")
    parser.add_argument("--teacher-z-window-mm", type=float, default=8.0)
    parser.add_argument("--teacher-z-refine-step-mm", type=float, default=2.0)
    parser.add_argument("--wide-xy-enable", action="store_true")
    parser.add_argument("--wide-xy-dx", type=int, default=16)
    parser.add_argument("--wide-xy-dy-min", type=int, default=-6)
    parser.add_argument("--wide-xy-dy-max", type=int, default=26)
    parser.add_argument("--wide-xy-step", type=int, default=4)
    parser.add_argument("--wide-xy-weight", type=float, default=0.78)
    parser.add_argument("--wide-xy-select-enable", action="store_true")
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
    args = parser.parse_args()
    process_dataset(
        args.data_dir,
        args.output_dir,
        MultiBoneConfig(
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
            teacher_low_z_enable=args.teacher_low_z_enable,
            teacher_low_z_range_mm=args.teacher_low_z_range_mm,
            teacher_low_z_min_shift_mm=args.teacher_low_z_min_shift_mm,
            teacher_low_z_step_mm=args.teacher_low_z_step_mm,
            teacher_low_z_weight=args.teacher_low_z_weight,
            teacher_low_z_select_enable=args.teacher_low_z_select_enable,
            contact_z_enable=args.contact_z_enable,
            contact_z_range_mm=args.contact_z_range_mm,
            contact_z_min_shift_mm=args.contact_z_min_shift_mm,
            contact_z_step_mm=args.contact_z_step_mm,
            contact_z_weight=args.contact_z_weight,
            contact_z_max_bone_fraction=args.contact_z_max_bone_fraction,
            contact_z_select_enable=args.contact_z_select_enable,
            teacher_z_refine_enable=args.teacher_z_refine_enable,
            teacher_z_window_mm=args.teacher_z_window_mm,
            teacher_z_refine_step_mm=args.teacher_z_refine_step_mm,
            wide_xy_enable=args.wide_xy_enable,
            wide_xy_dx=args.wide_xy_dx,
            wide_xy_dy_min=args.wide_xy_dy_min,
            wide_xy_dy_max=args.wide_xy_dy_max,
            wide_xy_step=args.wide_xy_step,
            wide_xy_weight=args.wide_xy_weight,
            wide_xy_select_enable=args.wide_xy_select_enable,
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
        ),
        teacher_csv=args.teacher_csv,
        case_names=set(args.cases) if args.cases else None,
        selection_policy=args.selection_policy,
        export_candidates=args.export_candidates,
        candidate_preview_topk=args.candidate_preview_topk,
        bone_mask_dir=args.bone_mask_dir,
        bone_mask_filename=args.bone_mask_filename,
        allow_threshold_bone_fallback=args.allow_threshold_bone_fallback,
        min_external_bone_voxels=args.min_external_bone_voxels,
    )
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
