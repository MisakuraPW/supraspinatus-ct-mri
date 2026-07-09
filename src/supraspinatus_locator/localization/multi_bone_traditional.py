from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from supraspinatus_locator.data.nifti_io import NiftiImage, load_nifti, save_nifti_like
from supraspinatus_locator.localization.roi_geometry import BBox3D, bbox_from_mask, bbox_iou, mask_from_bbox
from supraspinatus_locator.preprocessing.totalseg_bones import load_mask_compatible


@dataclass
class Component2D:
    area: int
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    fill: float
    aspect: float


@dataclass
class MultiBonePrediction:
    center_xyz: tuple[float, float, float]
    score: float
    lateral_sign: int
    humerus_anchor: Component2D
    roof_anchor: Component2D
    roi_stats: tuple[float, float, float, float, float, float, float, float, float]
    global_bone_centroid_x: float
    roi_centers_xyz: tuple[tuple[float, float, float], ...] | None = None
    roi_half_size: tuple[int, int, int] | None = None
    continuity_score: float = 0.0
    candidate_source: str = "current_multibone"
    anatomical_score: float = 0.0
    soft_tissue_score: float = 0.0
    safety_score: float = 0.0
    bone_distance_band_score: float = 0.0
    teacher_distance_mm: float | None = None
    arc_angle_deg: float | None = None
    surface_offset_voxels: float | None = None
    radius_normalized_distance: float | None = None
    same_anchor_support_count: int = 0
    edge_angle_deg: float | None = None
    edge_point_xyz: tuple[float, float, float] | None = None
    surface_normal_xy: tuple[float, float] | None = None
    surface_distance_mm: float | None = None
    cortical_edge_support: float | None = None
    soft_tissue_band_mean: float | None = None
    soft_tissue_band_std: float | None = None
    bone_edge_continuity_score: float | None = None
    arc_fit_residual: float | None = None
    bone_edge_tendon_score: float | None = None
    arc_fit_failed: bool = False


@dataclass
class MultiBoneConfig:
    bone_threshold_hu: float = 300.0
    half_size: tuple[int, int, int] = (22, 8, 2)
    z_range_fraction: tuple[float, float] = (0.35, 0.68)
    bone_margin_voxels: int = 3
    continuity_window: int = 2
    continuity_xy_tolerance: float = 14.0
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    top_k: int = 5
    current_anchor_count: int = 160
    low_z_enable: bool = False
    low_z_range_mm: float = 12.0
    low_z_step_mm: float = 2.0
    low_z_weight: float = 0.85
    branch_anchor_count: int = 6
    teacher_low_z_enable: bool = False
    teacher_low_z_range_mm: float = 16.0
    teacher_low_z_min_shift_mm: float = 4.0
    teacher_low_z_step_mm: float = 2.0
    teacher_low_z_weight: float = 0.88
    teacher_low_z_select_enable: bool = False
    contact_z_enable: bool = False
    contact_z_range_mm: float = 14.0
    contact_z_min_shift_mm: float = 6.0
    contact_z_step_mm: float = 2.0
    contact_z_weight: float = 0.68
    contact_z_max_bone_fraction: float = 0.12
    contact_z_select_enable: bool = False
    teacher_z_refine_enable: bool = False
    teacher_z_center: float | None = None
    teacher_center_xyz: tuple[float, float, float] | None = None
    teacher_z_window_mm: float = 8.0
    teacher_z_refine_step_mm: float = 2.0
    wide_xy_enable: bool = False
    wide_xy_dx: int = 16
    wide_xy_dy_min: int = -6
    wide_xy_dy_max: int = 26
    wide_xy_step: int = 4
    wide_xy_weight: float = 0.78
    wide_xy_select_enable: bool = False
    surface_arc_enable: bool = False
    surface_arc_select_enable: bool = False
    surface_arc_anchor_count: int = 8
    surface_arc_weight: float = 0.92
    surface_arc_angle_min_deg: float = 25.0
    surface_arc_angle_max_deg: float = 82.0
    surface_arc_angle_step_deg: float = 14.0
    surface_arc_offset_min_voxels: int = 4
    surface_arc_offset_max_voxels: int = 18
    surface_arc_offset_step_voxels: int = 4
    surface_arc_z_window: int = 1
    surface_arc_max_bone_fraction: float = 0.12
    surface_arc_target_bone_fraction: float = 0.035
    surface_arc_bone_sigma: float = 0.045
    surface_arc_target_offset_voxels: float = 10.0
    surface_arc_offset_sigma_voxels: float = 7.0
    surface_arc_sphere_blend: float = 0.25
    surface_arc_sphere_anchor_count: int = 24
    surface_arc_centerline_enable: bool = False
    surface_arc_centerline_points: int = 3
    surface_arc_centerline_angle_step_deg: float = 8.0
    surface_arc_centerline_half_size: tuple[int, int, int] = (14, 6, 1)
    bone_edge_enable: bool = False
    bone_edge_anchor_count: int = 8
    bone_edge_weight: float = 0.98
    bone_edge_angle_min_deg: float = 25.0
    bone_edge_angle_max_deg: float = 95.0
    bone_edge_angle_step_deg: float = 10.0
    bone_edge_offset_min_voxels: int = 4
    bone_edge_offset_max_voxels: int = 16
    bone_edge_offset_step_voxels: int = 4
    bone_edge_z_window: int = 1
    bone_edge_max_bone_fraction: float = 0.12
    bone_edge_target_bone_fraction: float = 0.025
    bone_edge_centerline_enable: bool = False
    bone_edge_centerline_points: int = 3
    bone_edge_centerline_angle_step_deg: float = 8.0
    bone_edge_centerline_half_size: tuple[int, int, int] = (14, 6, 1)
    bone_edge_channel_downshift_voxels: float = 0.0
    bone_dist_min_mm: float = 1.0
    bone_dist_good_min_mm: float = 3.0
    bone_dist_good_max_mm: float = 8.0
    bone_dist_far_mm: float = 15.0
    bone_dist_band_weight: float = 1.0


def connected_components_2d(mask: np.ndarray, min_area: int = 80) -> list[Component2D]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[Component2D] = []

    for sy, sx in np.argwhere(mask):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if len(points) < min_area:
            continue
        arr = np.asarray(points)
        y1, x1 = arr.min(axis=0)
        y2, x2 = arr.max(axis=0)
        cy, cx = arr.mean(axis=0)
        box_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
        components.append(
            Component2D(
                area=len(points),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                cx=float(cx),
                cy=float(cy),
                fill=float(len(points) / box_area),
                aspect=float((x2 - x1 + 1) / max(1, y2 - y1 + 1)),
            )
        )
    return components


def component_from_xy(points_x: np.ndarray, points_y: np.ndarray) -> Component2D:
    x1 = int(points_x.min())
    x2 = int(points_x.max())
    y1 = int(points_y.min())
    y2 = int(points_y.max())
    box_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
    return Component2D(
        area=int(len(points_x)),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        cx=float(points_x.mean()),
        cy=float(points_y.mean()),
        fill=float(len(points_x) / box_area),
        aspect=float((x2 - x1 + 1) / max(1, y2 - y1 + 1)),
    )


def proximal_humerus_anchor_component(bone_slice_yx: np.ndarray, comp: Component2D, lateral_sign: int) -> Component2D:
    local = bone_slice_yx[comp.y1 : comp.y2 + 1, comp.x1 : comp.x2 + 1]
    points_y, points_x = np.nonzero(local)
    if len(points_x) == 0:
        return comp
    points_x = points_x + comp.x1
    points_y = points_y + comp.y1
    width = comp.x2 - comp.x1 + 1
    height = comp.y2 - comp.y1 + 1
    upper_height = max(35, min(85, int(height * 0.46)))
    lateral_width = max(38, min(85, int(width * 0.58)))
    upper_limit = comp.y1 + upper_height
    if lateral_sign < 0:
        lateral_limit = comp.x1 + lateral_width
        anchor_mask = (points_y <= upper_limit) & (points_x <= lateral_limit)
    else:
        lateral_limit = comp.x2 - lateral_width
        anchor_mask = (points_y <= upper_limit) & (points_x >= lateral_limit)
    if int(anchor_mask.sum()) < 80:
        anchor_mask = points_y <= upper_limit
    if int(anchor_mask.sum()) < 80:
        return comp
    return component_from_xy(points_x[anchor_mask], points_y[anchor_mask])


def find_humerus_components(components: list[Component2D], image_shape: tuple[int, int, int], lateral_sign: int) -> list[Component2D]:
    out = []
    for comp in components:
        width = comp.x2 - comp.x1 + 1
        height = comp.y2 - comp.y1 + 1
        if not (
            700 <= comp.area <= 10000
            and 45 <= width <= 170
            and 65 <= height <= 240
            and 0.30 <= comp.aspect <= 1.65
            and comp.y2 > 250
            and 165 <= comp.y1 <= 245
            and 0.08 <= comp.fill <= 0.60
        ):
            continue
        if lateral_sign < 0 and comp.cx > image_shape[0] * 0.62:
            continue
        if lateral_sign > 0 and comp.cx < image_shape[0] * 0.38:
            continue
        out.append(comp)
    return out


def find_roof_components(components: list[Component2D], humerus_anchor: Component2D, lateral_sign: int) -> list[Component2D]:
    """Find superior bone components consistent with the acromion/scapular roof.

    In some cases the roof appears as a thin separate component. In others it is
    connected to a larger scapular component, so this intentionally accepts
    broader superior components and estimates the local undersurface later.
    """

    out = []
    humerus_edge = humerus_anchor.x1 if lateral_sign < 0 else humerus_anchor.x2
    for comp in components:
        width = comp.x2 - comp.x1 + 1
        height = comp.y2 - comp.y1 + 1
        raw_gap = humerus_anchor.y1 - comp.y2
        overlaps_anchor_y = comp.y1 < humerus_anchor.y1 and comp.y2 >= humerus_anchor.y1 - 18
        if not (
            180 <= comp.area <= 18000
            and 35 <= width <= 380
            and 5 <= height <= 180
            and comp.aspect >= 0.85
            and (-25 <= raw_gap <= 105 or overlaps_anchor_y)
            and comp.y1 < humerus_anchor.y1
        ):
            continue
        if lateral_sign > 0:
            overlap_ok = comp.x2 >= humerus_anchor.x1 - 80 and comp.x1 <= humerus_edge
        else:
            overlap_ok = comp.x1 <= humerus_anchor.x2 + 80 and comp.x2 >= humerus_edge
        if overlap_ok:
            out.append(comp)
    return out


def roof_undersurface_y(roof: Component2D, humerus_anchor: Component2D) -> float:
    if roof.y2 < humerus_anchor.y1:
        return float(roof.y2)
    height = roof.y2 - roof.y1 + 1
    local_under = roof.y1 + min(38, max(14, int(height * 0.45)))
    return float(min(humerus_anchor.y1 - 6, local_under))


def estimate_humeral_head_circle(humerus: Component2D, humerus_anchor: Component2D) -> tuple[float, float, float]:
    """Approximate the humeral head circle in the current 2D CT slice.

    The component can include shaft voxels, so the center is biased toward the
    superior part of the component rather than the full component centroid.
    This is intentionally coarse; it is used to create surface-following
    candidate ROIs, not as a final segmentation of the humeral head.
    """

    width = humerus.x2 - humerus.x1 + 1
    height = humerus.y2 - humerus.y1 + 1
    anchor_width = humerus_anchor.x2 - humerus_anchor.x1 + 1
    anchor_height = humerus_anchor.y2 - humerus_anchor.y1 + 1
    radius = float(max(24.0, min(72.0, 0.48 * max(anchor_width, anchor_height), 0.42 * min(width, height))))
    center_x = float(humerus_anchor.cx)
    center_y = float(humerus.y1 + min(height * 0.42, radius * 1.35 + 8.0))
    return center_x, center_y, radius


def humeral_head_likeness(humerus: Component2D, humerus_anchor: Component2D) -> float:
    """Score whether a candidate bone component looks like a humeral head.

    The roof relationship alone can prefer the glenoid/scapular side in some
    slices. This term keeps a component in play only when the bone itself has
    a plausible round proximal humerus footprint and a non-narrow upper-lateral
    anchor.
    """

    width = float(humerus.x2 - humerus.x1 + 1)
    height = float(humerus.y2 - humerus.y1 + 1)
    anchor_width = float(humerus_anchor.x2 - humerus_anchor.x1 + 1)
    anchor_height = float(humerus_anchor.y2 - humerus_anchor.y1 + 1)
    size_prior = float(np.exp(-((width - 105.0) / 45.0) ** 2) * np.exp(-((height - 105.0) / 52.0) ** 2))
    anchor_prior = float(np.exp(-((anchor_width - 64.0) / 22.0) ** 2) * np.exp(-((anchor_height - 50.0) / 20.0) ** 2))
    aspect_prior = float(np.exp(-((np.log(max(width, 1.0) / max(height, 1.0))) / 0.48) ** 2))
    fill_prior = float(np.exp(-((float(humerus.fill) - 0.24) / 0.18) ** 2))
    non_narrow = float(min(1.0, anchor_width / 58.0) * min(1.0, anchor_height / 42.0))
    return float((0.42 * size_prior + 0.34 * anchor_prior + 0.18 * aspect_prior + 0.06 * fill_prior) * non_narrow)


def erode_binary_2d(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion

        return binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    except Exception:
        padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
        out = np.ones(mask.shape, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return out


def fit_circle_xy(points_x: np.ndarray, points_y: np.ndarray) -> tuple[float, float, float, float] | None:
    if len(points_x) < 12:
        return None
    x = points_x.astype(float)
    y = points_y.astype(float)
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = -float(sol[0]) / 2.0
    cy = -float(sol[1]) / 2.0
    radius_sq = cx * cx + cy * cy - float(sol[2])
    if radius_sq <= 1.0:
        return None
    radius = float(np.sqrt(radius_sq))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = float(np.median(np.abs(d - radius)) / max(radius, 1e-6))
    return cx, cy, radius, residual


def component_edge_points(
    bone_slice_yx: np.ndarray,
    comp: Component2D,
    lateral_sign: int,
    angle_min_deg: float,
    angle_max_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = bone_slice_yx[comp.y1 : comp.y2 + 1, comp.x1 : comp.x2 + 1].astype(bool)
    edge = local & ~erode_binary_2d(local)
    points_y, points_x = np.nonzero(edge)
    if len(points_x) == 0:
        return points_x, points_y, np.asarray([], dtype=float), np.asarray([], dtype=float)
    points_x = points_x.astype(float) + comp.x1
    points_y = points_y.astype(float) + comp.y1
    rough_cx, rough_cy, _rough_r = estimate_humeral_head_circle(comp, proximal_humerus_anchor_component(bone_slice_yx, comp, lateral_sign))
    dx = points_x - rough_cx
    dy = points_y - rough_cy
    angles = np.degrees(np.arctan2(-dy, dx * lateral_sign))
    keep = (
        (angles >= angle_min_deg)
        & (angles <= angle_max_deg)
        & (dx * lateral_sign >= -4.0)
        & (dy <= 12.0)
    )
    return points_x[keep], points_y[keep], angles[keep], keep


def soft_tissue_band_stats(
    image: np.ndarray,
    point_x: float,
    point_y: float,
    z: int,
    normal_x: float,
    normal_y: float,
    offsets: range,
    band_half_width: int = 3,
) -> tuple[float, float]:
    samples: list[float] = []
    tangent_x = -normal_y
    tangent_y = normal_x
    for offset in offsets:
        cx = point_x + normal_x * offset
        cy = point_y + normal_y * offset
        for t in range(-band_half_width, band_half_width + 1):
            x = int(round(cx + tangent_x * t))
            y = int(round(cy + tangent_y * t))
            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and 0 <= z < image.shape[2]:
                samples.append(float(image[x, y, z]))
    if not samples:
        return 0.0, 0.0
    arr = np.asarray(samples, dtype=float)
    return float(arr.mean()), float(arr.std())


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        return [start]
    out = []
    value = start
    while value <= stop + 1e-6:
        out.append(float(value))
        value += step
    return out


def box_from_center(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    half_size: tuple[int, int, int],
) -> BBox3D:
    x, y, z = center
    hx, hy, hz = half_size
    return BBox3D(
        (
            max(0, int(round(x - hx))),
            max(0, int(round(y - hy))),
            max(0, int(round(z - hz))),
        ),
        (
            min(shape[0] - 1, int(round(x + hx))),
            min(shape[1] - 1, int(round(y + hy))),
            min(shape[2] - 1, int(round(z + hz))),
        ),
    )


def expand_box(box: BBox3D, shape: tuple[int, int, int], margin: int) -> BBox3D:
    return BBox3D(
        (
            max(0, box.min[0] - margin),
            max(0, box.min[1] - margin),
            max(0, box.min[2] - margin),
        ),
        (
            min(shape[0] - 1, box.max[0] + margin),
            min(shape[1] - 1, box.max[1] + margin),
            min(shape[2] - 1, box.max[2] + margin),
        ),
    )


def box_volume(box: BBox3D) -> int:
    return (box.max[0] - box.min[0] + 1) * (box.max[1] - box.min[1] + 1) * (box.max[2] - box.min[2] + 1)


def box_bone_count(bone_mask: np.ndarray, box: BBox3D) -> int:
    x1, y1, z1 = box.min
    x2, y2, z2 = box.max
    return int(bone_mask[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1].sum())


def bone_shell_fraction(bone_mask: np.ndarray, box: BBox3D, margin: int) -> float:
    expanded = expand_box(box, bone_mask.shape, margin)
    shell_volume = box_volume(expanded) - box_volume(box)
    if shell_volume <= 0:
        return 0.0
    shell_bone = box_bone_count(bone_mask, expanded) - box_bone_count(bone_mask, box)
    return float(shell_bone / shell_volume)


def min_box_to_bone_distance_mm(bone_mask: np.ndarray, box: BBox3D, spacing: np.ndarray, max_distance_mm: float) -> float:
    if box_bone_count(bone_mask, box) > 0:
        return 0.0
    max_margin = int(np.ceil(max_distance_mm / max(float(np.min(spacing)), 1e-6))) + 2
    expanded = expand_box(box, bone_mask.shape, max_margin)
    x1, y1, z1 = expanded.min
    x2, y2, z2 = expanded.max
    local_points = np.argwhere(bone_mask[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1])
    if len(local_points) == 0:
        return float(max_distance_mm + 1.0)
    points = local_points + np.asarray([x1, y1, z1])
    mins = np.asarray(box.min)
    maxs = np.asarray(box.max)
    low_gap = np.maximum(mins - points, 0)
    high_gap = np.maximum(points - maxs, 0)
    gaps = (low_gap + high_gap).astype(float) * spacing
    return float(np.sqrt(np.sum(gaps * gaps, axis=1)).min())


def approximate_box_to_bone_distance_mm(
    bone_mask: np.ndarray,
    box: BBox3D,
    spacing: np.ndarray,
    roi_bone_fraction: float,
    cfg: MultiBoneConfig,
) -> float:
    if roi_bone_fraction > 0:
        return 0.0
    min_spacing = max(float(np.min(spacing)), 1e-6)
    checks = [
        (cfg.bone_dist_min_mm, max(1, int(np.ceil(cfg.bone_dist_min_mm / min_spacing)))),
        (cfg.bone_dist_good_min_mm, max(1, int(np.ceil(cfg.bone_dist_good_min_mm / min_spacing)))),
        (cfg.bone_dist_good_max_mm, max(1, int(np.ceil(cfg.bone_dist_good_max_mm / min_spacing)))),
        (cfg.bone_dist_far_mm, max(1, int(np.ceil(cfg.bone_dist_far_mm / min_spacing)))),
    ]
    previous_margin = 0
    previous_count = box_bone_count(bone_mask, box)
    for distance_mm, margin_voxels in checks:
        expanded = expand_box(box, bone_mask.shape, margin_voxels)
        count = box_bone_count(bone_mask, expanded)
        if count > previous_count:
            return float(distance_mm)
        previous_margin = margin_voxels
        previous_count = count
    return float(cfg.bone_dist_far_mm + min_spacing)


def bone_distance_band_score(distance_mm: float, cfg: MultiBoneConfig) -> float:
    if distance_mm <= 0:
        return -0.8
    if distance_mm < cfg.bone_dist_min_mm:
        return -0.55
    if distance_mm < cfg.bone_dist_good_min_mm:
        return 0.05
    if distance_mm <= cfg.bone_dist_good_max_mm:
        center = (cfg.bone_dist_good_min_mm + cfg.bone_dist_good_max_mm) / 2.0
        half_width = max(1e-6, (cfg.bone_dist_good_max_mm - cfg.bone_dist_good_min_mm) / 2.0)
        return float(0.65 - min(1.0, abs(distance_mm - center) / half_width) * 0.20)
    if distance_mm <= cfg.bone_dist_far_mm:
        return -0.15 * (distance_mm - cfg.bone_dist_good_max_mm) / max(1e-6, cfg.bone_dist_far_mm - cfg.bone_dist_good_max_mm)
    return -0.8


def center_to_box_gap_mm(center: np.ndarray, box: BBox3D, spacing: np.ndarray) -> float:
    mins = np.asarray(box.min, dtype=float)
    maxs = np.asarray(box.max, dtype=float)
    low_gap = np.maximum(mins - center, 0.0)
    high_gap = np.maximum(center - maxs, 0.0)
    return float(np.linalg.norm((low_gap + high_gap) * spacing))


def box_to_box_gap_mm(a: BBox3D, b: BBox3D, spacing: np.ndarray) -> float:
    gaps = []
    for axis in range(3):
        a1, a2 = a.min[axis], a.max[axis]
        b1, b2 = b.min[axis], b.max[axis]
        if a2 < b1:
            gaps.append((b1 - a2) * spacing[axis])
        elif b2 < a1:
            gaps.append((a1 - b2) * spacing[axis])
        else:
            gaps.append(0.0)
    return float(np.linalg.norm(np.asarray(gaps, dtype=float)))


def roi_stats(
    image: np.ndarray,
    bone_mask: np.ndarray,
    center: tuple[float, float, float],
    half_size: tuple[int, int, int],
    cfg: MultiBoneConfig,
) -> tuple[float, float, float, float, float, float, float, float, float] | None:
    bbox = box_from_center(image.shape, center, half_size)
    x1, y1, z1 = bbox.min
    x2, y2, z2 = bbox.max
    roi = image[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
    roi_bone = bone_mask[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
    if roi.size == 0:
        return None
    by1 = min(image.shape[1], y2 + 2)
    by2 = min(image.shape[1], y2 + 45)
    ay1 = max(0, y1 - 65)
    ay2 = max(0, y1 - 10)
    below = float(bone_mask[x1 : x2 + 1, by1:by2, z1 : z2 + 1].mean()) if by2 > by1 else 0.0
    above = float(bone_mask[x1 : x2 + 1, ay1:ay2, z1 : z2 + 1].mean()) if ay2 > ay1 else 0.0
    roi_bone_fraction = float(roi_bone.mean())
    bone_distance_mm = approximate_box_to_bone_distance_mm(bone_mask, bbox, np.asarray(cfg.spacing_xyz, dtype=float), roi_bone_fraction, cfg)
    return (
        roi_bone_fraction,
        float(roi.mean()),
        below,
        above,
        float((roi > -200).mean()),
        bone_shell_fraction(bone_mask, bbox, 1),
        bone_shell_fraction(bone_mask, bbox, max(1, cfg.bone_margin_voxels)),
        bone_distance_mm,
        bone_distance_band_score(bone_distance_mm, cfg),
    )


def mask_from_centers(shape: tuple[int, int, int], centers: tuple[tuple[float, float, float], ...], half_size: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for center in centers:
        mask |= mask_from_bbox(shape, box_from_center(shape, center, half_size)).astype(bool)
    return mask


def roi_stats_from_mask(
    image: np.ndarray,
    bone_mask: np.ndarray,
    mask: np.ndarray,
    cfg: MultiBoneConfig,
) -> tuple[float, float, float, float, float, float, float, float, float] | None:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    values = image[mask > 0]
    if values.size == 0:
        return None
    x1, y1, z1 = bbox.min
    x2, y2, z2 = bbox.max
    by1 = min(image.shape[1], y2 + 2)
    by2 = min(image.shape[1], y2 + 45)
    ay1 = max(0, y1 - 65)
    ay2 = max(0, y1 - 10)
    below = float(bone_mask[x1 : x2 + 1, by1:by2, z1 : z2 + 1].mean()) if by2 > by1 else 0.0
    above = float(bone_mask[x1 : x2 + 1, ay1:ay2, z1 : z2 + 1].mean()) if ay2 > ay1 else 0.0
    roi_bone_fraction = float(bone_mask[mask > 0].mean())
    bone_distance_mm = approximate_box_to_bone_distance_mm(
        bone_mask,
        bbox,
        np.asarray(cfg.spacing_xyz, dtype=float),
        roi_bone_fraction,
        cfg,
    )
    return (
        roi_bone_fraction,
        float(values.mean()),
        below,
        above,
        float((values > -200).mean()),
        bone_shell_fraction(bone_mask, bbox, 1),
        bone_shell_fraction(bone_mask, bbox, max(1, cfg.bone_margin_voxels)),
        bone_distance_mm,
        bone_distance_band_score(bone_distance_mm, cfg),
    )


def roi_stats_from_centers(
    image: np.ndarray,
    bone_mask: np.ndarray,
    centers: tuple[tuple[float, float, float], ...],
    half_size: tuple[int, int, int],
    cfg: MultiBoneConfig,
) -> tuple[float, float, float, float, float, float, float, float, float] | None:
    boxes = [box_from_center(image.shape, center, half_size) for center in centers]
    if not boxes:
        return None
    bbox = BBox3D(
        (
            min(box.min[0] for box in boxes),
            min(box.min[1] for box in boxes),
            min(box.min[2] for box in boxes),
        ),
        (
            max(box.max[0] for box in boxes),
            max(box.max[1] for box in boxes),
            max(box.max[2] for box in boxes),
        ),
    )
    x1, y1, z1 = bbox.min
    x2, y2, z2 = bbox.max
    local_mask = np.zeros((x2 - x1 + 1, y2 - y1 + 1, z2 - z1 + 1), dtype=bool)
    for box in boxes:
        local_mask[
            box.min[0] - x1 : box.max[0] - x1 + 1,
            box.min[1] - y1 : box.max[1] - y1 + 1,
            box.min[2] - z1 : box.max[2] - z1 + 1,
        ] = True
    local_image = image[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
    local_bone = bone_mask[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
    values = local_image[local_mask]
    if values.size == 0:
        return None
    by1 = min(image.shape[1], y2 + 2)
    by2 = min(image.shape[1], y2 + 45)
    ay1 = max(0, y1 - 65)
    ay2 = max(0, y1 - 10)
    below = float(bone_mask[x1 : x2 + 1, by1:by2, z1 : z2 + 1].mean()) if by2 > by1 else 0.0
    above = float(bone_mask[x1 : x2 + 1, ay1:ay2, z1 : z2 + 1].mean()) if ay2 > ay1 else 0.0
    roi_bone_fraction = float(local_bone[local_mask].mean())
    bone_distance_mm = approximate_box_to_bone_distance_mm(
        bone_mask,
        bbox,
        np.asarray(cfg.spacing_xyz, dtype=float),
        roi_bone_fraction,
        cfg,
    )
    return (
        roi_bone_fraction,
        float(values.mean()),
        below,
        above,
        float((values > -200).mean()),
        bone_shell_fraction(bone_mask, bbox, 1),
        bone_shell_fraction(bone_mask, bbox, max(1, cfg.bone_margin_voxels)),
        bone_distance_mm,
        bone_distance_band_score(bone_distance_mm, cfg),
    )


def anchor_continuity_score(anchor: dict[str, object], anchors: list[dict[str, object]], cfg: MultiBoneConfig) -> float:
    support = 0.0
    for other in anchors:
        dz = abs(int(anchor["z"]) - int(other["z"]))
        if dz == 0 or dz > cfg.continuity_window:
            continue
        base_dist = float(
            np.sqrt(
                (float(anchor["base_x"]) - float(other["base_x"])) ** 2
                + (float(anchor["base_y"]) - float(other["base_y"])) ** 2
            )
        )
        gap_dist = abs(float(anchor["gap"]) - float(other["gap"]))
        if base_dist <= cfg.continuity_xy_tolerance and gap_dist <= 18.0:
            support += 1.0 / dz
    max_support = sum(2.0 / dz for dz in range(1, cfg.continuity_window + 1))
    return float(min(1.0, support / max_support)) if max_support > 0 else 0.0


def anchor_support_count(anchor: dict[str, object], anchors: list[dict[str, object]], cfg: MultiBoneConfig) -> int:
    count = 0
    for other in anchors:
        dz = abs(int(anchor["z"]) - int(other["z"]))
        if dz == 0 or dz > cfg.continuity_window:
            continue
        base_dist = float(
            np.sqrt(
                (float(anchor["base_x"]) - float(other["base_x"])) ** 2
                + (float(anchor["base_y"]) - float(other["base_y"])) ** 2
            )
        )
        gap_dist = abs(float(anchor["gap"]) - float(other["gap"]))
        if base_dist <= cfg.continuity_xy_tolerance and gap_dist <= 18.0:
            count += 1
    return count


def radius_normalized_distance(center: tuple[float, float, float], anchor: dict[str, object]) -> float:
    radius = max(1e-6, float(anchor["humerus_circle_radius"]))
    dx = float(center[0]) - float(anchor["humerus_circle_x"])
    dy = float(center[1]) - float(anchor["humerus_circle_y"])
    return float(np.sqrt(dx * dx + dy * dy) / radius)


def locate_multibone_candidates(
    image: np.ndarray,
    config: MultiBoneConfig | None = None,
    top_k: int | None = None,
    bone_mask_override: np.ndarray | None = None,
) -> list[MultiBonePrediction]:
    cfg = config or MultiBoneConfig()
    bone_mask = np.asarray(bone_mask_override, dtype=bool) if bone_mask_override is not None else image > cfg.bone_threshold_hu
    if bone_mask.shape != image.shape:
        raise ValueError(f"bone_mask_override shape {bone_mask.shape} does not match image shape {image.shape}")
    bone_points = np.argwhere(bone_mask)
    if len(bone_points) == 0:
        raise ValueError("No bone voxels found with the configured bone mask")

    global_bone_centroid_x = float(bone_points[:, 0].mean())
    lateral_sign = -1 if global_bone_centroid_x > image.shape[0] / 2 else 1
    candidates: list[MultiBonePrediction] = []

    z_start = max(1, int(image.shape[2] * cfg.z_range_fraction[0]))
    z_end = min(image.shape[2] - 1, int(image.shape[2] * cfg.z_range_fraction[1]))
    hx, hy, hz = cfg.half_size
    anchor_models: list[dict[str, object]] = []

    for z in range(z_start, z_end):
        bone_slice = bone_mask[:, :, z].T
        if int(bone_slice.sum()) < 500:
            continue
        components = connected_components_2d(bone_slice, min_area=80)
        for humerus in find_humerus_components(components, image.shape, lateral_sign):
            humerus_anchor = proximal_humerus_anchor_component(bone_slice, humerus, lateral_sign)
            roofs = find_roof_components(components, humerus_anchor, lateral_sign)
            if not roofs:
                continue

            for roof in roofs:
                roof_y = roof_undersurface_y(roof, humerus_anchor)
                circle_x, circle_y, circle_radius = estimate_humeral_head_circle(humerus, humerus_anchor)
                head_prior = humeral_head_likeness(humerus, humerus_anchor)
                if lateral_sign > 0:
                    tendon_base_x = humerus_anchor.x2 - hx - 2
                    roof_edge_x = roof.x2
                else:
                    tendon_base_x = humerus_anchor.x1 + hx + 2
                    roof_edge_x = roof.x1
                gap = humerus_anchor.y1 - roof_y
                base_x = (tendon_base_x + roof_edge_x) / 2.0
                if gap > 48:
                    base_y = roof_y + min(18.0, gap * 0.18)
                else:
                    base_y = (roof_y + humerus_anchor.y1) / 2.0 + gap * 0.16

                roof_width = roof.x2 - roof.x1 + 1
                roof_height = roof.y2 - roof.y1 + 1
                roof_prior = np.exp(-((roof.aspect - 6.0) / 3.5) ** 2)
                roof_fill_prior = np.exp(-((roof.fill - 0.55) / 0.35) ** 2)
                gap_prior = np.exp(-((gap - 18.0) / 10.0) ** 2)
                z_prior = np.exp(-((z / image.shape[2] - 0.515) / 0.075) ** 2)
                width_prior = np.exp(-((roof_width - 120.0) / 70.0) ** 2)
                height_prior = np.exp(-((roof_height - 18.0) / 18.0) ** 2)
                anchor_models.append(
                    {
                        "z": z,
                        "humerus_full": humerus,
                        "humerus_anchor": humerus_anchor,
                        "roof": roof,
                        "humerus_circle_x": circle_x,
                        "humerus_circle_y": circle_y,
                        "humerus_circle_radius": circle_radius,
                        "base_x": base_x,
                        "base_y": base_y,
                        "gap": gap,
                        "roof_prior": roof_prior,
                        "roof_fill_prior": roof_fill_prior,
                        "gap_prior": gap_prior,
                        "z_prior": z_prior,
                        "width_prior": width_prior,
                        "height_prior": height_prior,
                        "head_prior": head_prior,
                        "anchor_score": (
                            roof_prior * 1.05
                            + roof_fill_prior * 0.30
                            + gap_prior * 0.85
                            + z_prior * 0.70
                            + width_prior * 0.30
                            + height_prior * 0.20
                            + head_prior * 1.45
                        ),
                    }
                )

    def add_anchor_candidates(
        anchor: dict[str, object],
        z_values: list[int],
        source: str,
        source_weight: float,
        dx_values: range | None = None,
        dy_values: range | None = None,
        max_bone_fraction: float = 0.012,
        bone_penalty_weight: float = 45.0,
        near_bone_penalty_weight: float = 0.9,
        margin_bone_penalty_weight: float = 0.45,
    ) -> None:
        continuity = anchor_continuity_score(anchor, anchor_models, cfg)
        support_count = anchor_support_count(anchor, anchor_models, cfg)
        if source == "current_multibone" and continuity <= 0.0:
            return
        humerus_anchor = anchor["humerus_anchor"]
        roof = anchor["roof"]
        base_x = float(anchor["base_x"])
        base_y = float(anchor["base_y"])
        teacher_center = np.asarray(cfg.teacher_center_xyz, dtype=float) if cfg.teacher_center_xyz is not None else None

        for z in z_values:
            if not (0 <= z < image.shape[2]):
                continue
            z_prior = float(np.exp(-((z / image.shape[2] - 0.515) / 0.075) ** 2))
            for dx in dx_values or range(-10, 11, 2):
                for dy in dy_values or range(-4, 9, 2):
                    center = (float(base_x + dx), float(base_y + dy), float(z))
                    stats = roi_stats(image, bone_mask, center, cfg.half_size, cfg)
                    if stats is None:
                        continue
                    (
                        bone_fraction,
                        mean_value,
                        below_bone,
                        above_bone,
                        body_fraction,
                        near_bone_fraction,
                        margin_bone_fraction,
                        _bone_distance_mm,
                        distance_band_score,
                    ) = stats
                    if bone_fraction > max_bone_fraction or body_fraction < 0.94 or not (-20 <= mean_value <= 140):
                        continue

                    soft_prior = float(np.exp(-((mean_value - 50.0) / 35.0) ** 2))
                    corridor_x_sigma = 16.0 if source == "wide_xy" else 12.0
                    corridor_y_sigma = 20.0 if source == "wide_xy" else 8.0
                    corridor_x_prior = float(np.exp(-((center[0] - base_x) / corridor_x_sigma) ** 2))
                    corridor_y_prior = float(np.exp(-((center[1] - base_y) / corridor_y_sigma) ** 2))
                    anatomical_score = (
                        float(anchor["roof_prior"]) * 1.25
                        + float(anchor["roof_fill_prior"]) * 0.35
                        + float(anchor["gap_prior"]) * 1.1
                        + z_prior * 0.75
                        + float(anchor["width_prior"]) * 0.35
                        + float(anchor["height_prior"]) * 0.25
                        + float(anchor.get("head_prior", 0.0)) * 0.45
                        + corridor_x_prior * 0.45
                        + corridor_y_prior * 0.55
                        + continuity * 0.55
                    )
                    soft_tissue_score = soft_prior * 0.8 + below_bone * 0.16 + above_bone * 0.08
                    safety_score = (
                        -bone_fraction * bone_penalty_weight
                        -near_bone_fraction * near_bone_penalty_weight
                        -margin_bone_fraction * margin_bone_penalty_weight
                        + distance_band_score * cfg.bone_dist_band_weight
                    )
                    teacher_distance_mm = None
                    if teacher_center is not None:
                        teacher_distance_mm = float(np.sqrt(np.sum(((np.asarray(center) - teacher_center) * np.asarray(cfg.spacing_xyz)) ** 2)))
                    score = (anatomical_score + soft_tissue_score + safety_score) * source_weight
                    candidates.append(
                        MultiBonePrediction(
                            center_xyz=center,
                            score=float(score),
                            lateral_sign=lateral_sign,
                            humerus_anchor=humerus_anchor,
                            roof_anchor=roof,
                            roi_stats=stats,
                            global_bone_centroid_x=global_bone_centroid_x,
                            continuity_score=continuity,
                            candidate_source=source,
                            anatomical_score=float(anatomical_score),
                            soft_tissue_score=float(soft_tissue_score),
                            safety_score=float(safety_score),
                            bone_distance_band_score=float(distance_band_score),
                            teacher_distance_mm=teacher_distance_mm,
                            radius_normalized_distance=radius_normalized_distance(center, anchor),
                            same_anchor_support_count=support_count,
                        )
                    )

    def add_surface_arc_candidates(anchor: dict[str, object], z_values: list[int]) -> None:
        continuity = anchor_continuity_score(anchor, anchor_models, cfg)
        support_count = anchor_support_count(anchor, anchor_models, cfg)
        if continuity <= 0.0:
            return
        humerus_anchor = anchor["humerus_anchor"]
        roof = anchor["roof"]
        base_x = float(anchor["base_x"])
        base_y = float(anchor["base_y"])
        circle_x = float(anchor["humerus_circle_x"])
        circle_y = float(anchor["humerus_circle_y"])
        circle_radius = float(anchor["humerus_circle_radius"])
        teacher_center = np.asarray(cfg.teacher_center_xyz, dtype=float) if cfg.teacher_center_xyz is not None else None

        angle_values = frange(cfg.surface_arc_angle_min_deg, cfg.surface_arc_angle_max_deg, cfg.surface_arc_angle_step_deg)
        offset_values = range(
            int(cfg.surface_arc_offset_min_voxels),
            int(cfg.surface_arc_offset_max_voxels) + 1,
            max(1, int(cfg.surface_arc_offset_step_voxels)),
        )
        for z in z_values:
            if not (0 <= z < image.shape[2]):
                continue
            z_prior = float(np.exp(-((z / image.shape[2] - 0.515) / 0.075) ** 2))
            for angle_deg in angle_values:
                theta = np.deg2rad(angle_deg)
                normal_x = float(lateral_sign * np.cos(theta))
                normal_y = float(-np.sin(theta))
                angle_prior = float(np.exp(-((angle_deg - 52.0) / 24.0) ** 2))
                for offset in offset_values:
                    center = (
                        float(circle_x + normal_x * (circle_radius + offset)),
                        float(circle_y + normal_y * (circle_radius + offset)),
                        float(z),
                    )
                    stats = roi_stats(image, bone_mask, center, cfg.half_size, cfg)
                    if stats is None:
                        continue
                    (
                        bone_fraction,
                        mean_value,
                        below_bone,
                        above_bone,
                        body_fraction,
                        near_bone_fraction,
                        margin_bone_fraction,
                        _bone_distance_mm,
                        distance_band_score,
                    ) = stats
                    if bone_fraction > cfg.surface_arc_max_bone_fraction or body_fraction < 0.94 or not (-20 <= mean_value <= 150):
                        continue

                    soft_prior = float(np.exp(-((mean_value - 55.0) / 42.0) ** 2))
                    corridor_x_prior = float(np.exp(-((center[0] - base_x) / 22.0) ** 2))
                    corridor_y_prior = float(np.exp(-((center[1] - base_y) / 20.0) ** 2))
                    offset_prior = float(
                        np.exp(-((float(offset) - cfg.surface_arc_target_offset_voxels) / max(1e-6, cfg.surface_arc_offset_sigma_voxels)) ** 2)
                    )
                    surface_contact_score = float(
                        np.exp(-((bone_fraction - cfg.surface_arc_target_bone_fraction) / max(1e-6, cfg.surface_arc_bone_sigma)) ** 2)
                    )
                    if bone_fraction <= 0.0:
                        surface_contact_score = max(surface_contact_score, min(0.75, margin_bone_fraction * 12.0))
                    anatomical_score = (
                        float(anchor["roof_prior"]) * 1.05
                        + float(anchor["roof_fill_prior"]) * 0.30
                        + float(anchor["gap_prior"]) * 0.95
                        + z_prior * 0.65
                        + float(anchor["width_prior"]) * 0.25
                        + float(anchor["height_prior"]) * 0.20
                        + float(anchor.get("head_prior", 0.0)) * 0.45
                        + corridor_x_prior * 0.35
                        + corridor_y_prior * 0.40
                        + continuity * 0.50
                        + angle_prior * 0.65
                        + offset_prior * 0.45
                        + surface_contact_score * 1.15
                    )
                    soft_tissue_score = soft_prior * 0.75 + below_bone * 0.12 + above_bone * 0.06
                    deep_bone_excess = max(0.0, bone_fraction - cfg.surface_arc_target_bone_fraction)
                    safety_score = (
                        -deep_bone_excess * 8.0
                        -max(0.0, near_bone_fraction - 0.10) * 0.40
                        -max(0.0, margin_bone_fraction - 0.14) * 0.18
                        + distance_band_score * cfg.bone_dist_band_weight * 0.25
                    )
                    teacher_distance_mm = None
                    if teacher_center is not None:
                        teacher_distance_mm = float(np.sqrt(np.sum(((np.asarray(center) - teacher_center) * np.asarray(cfg.spacing_xyz)) ** 2)))
                    roi_centers = None
                    roi_half_size = None
                    if cfg.surface_arc_centerline_enable:
                        point_count = max(1, int(cfg.surface_arc_centerline_points))
                        middle = (point_count - 1) / 2.0
                        centers = []
                        for point_index in range(point_count):
                            point_angle_deg = angle_deg + (point_index - middle) * cfg.surface_arc_centerline_angle_step_deg
                            point_theta = np.deg2rad(point_angle_deg)
                            point_normal_x = float(lateral_sign * np.cos(point_theta))
                            point_normal_y = float(-np.sin(point_theta))
                            centers.append(
                                (
                                    float(circle_x + point_normal_x * (circle_radius + offset)),
                                    float(circle_y + point_normal_y * (circle_radius + offset)),
                                    float(z),
                                )
                            )
                        roi_centers = tuple(centers)
                        roi_half_size = cfg.surface_arc_centerline_half_size
                    score = (anatomical_score + soft_tissue_score + safety_score) * cfg.surface_arc_weight
                    candidates.append(
                        MultiBonePrediction(
                            center_xyz=center,
                            score=float(score),
                            lateral_sign=lateral_sign,
                            humerus_anchor=humerus_anchor,
                            roof_anchor=roof,
                            roi_stats=stats,
                            global_bone_centroid_x=global_bone_centroid_x,
                            roi_centers_xyz=roi_centers,
                            roi_half_size=roi_half_size,
                            continuity_score=continuity,
                            candidate_source="surface_arc",
                            anatomical_score=float(anatomical_score),
                            soft_tissue_score=float(soft_tissue_score),
                            safety_score=float(safety_score),
                            bone_distance_band_score=float(distance_band_score),
                            teacher_distance_mm=teacher_distance_mm,
                            arc_angle_deg=float(angle_deg),
                            surface_offset_voxels=float(offset),
                            radius_normalized_distance=radius_normalized_distance(center, anchor),
                            same_anchor_support_count=support_count,
                        )
                    )

    def add_bone_edge_tendon_candidates(anchor: dict[str, object], z_values: list[int]) -> None:
        continuity = anchor_continuity_score(anchor, anchor_models, cfg)
        support_count = anchor_support_count(anchor, anchor_models, cfg)
        if continuity <= 0.0:
            return
        humerus = anchor["humerus_full"]
        humerus_anchor = anchor["humerus_anchor"]
        roof = anchor["roof"]
        teacher_center = np.asarray(cfg.teacher_center_xyz, dtype=float) if cfg.teacher_center_xyz is not None else None
        angle_values = frange(cfg.bone_edge_angle_min_deg, cfg.bone_edge_angle_max_deg, cfg.bone_edge_angle_step_deg)
        offset_values = range(
            int(cfg.bone_edge_offset_min_voxels),
            int(cfg.bone_edge_offset_max_voxels) + 1,
            max(1, int(cfg.bone_edge_offset_step_voxels)),
        )
        xy_spacing = float(np.mean(np.asarray(cfg.spacing_xyz[:2], dtype=float)))
        for z in z_values:
            if not (0 <= z < image.shape[2]):
                continue
            bone_slice = bone_mask[:, :, z].T
            edge_x, edge_y, edge_angles, _keep = component_edge_points(
                bone_slice,
                humerus,
                lateral_sign,
                cfg.bone_edge_angle_min_deg - 8.0,
                cfg.bone_edge_angle_max_deg + 8.0,
            )
            if len(edge_x) < 12:
                continue
            fit = fit_circle_xy(edge_x, edge_y)
            if fit is None:
                circle_x = float(anchor["humerus_circle_x"])
                circle_y = float(anchor["humerus_circle_y"])
                circle_radius = float(anchor["humerus_circle_radius"])
                arc_residual = 0.35
                arc_fit_failed = True
            else:
                circle_x, circle_y, circle_radius, arc_residual = fit
                if not (18.0 <= circle_radius <= 85.0):
                    circle_x = float(anchor["humerus_circle_x"])
                    circle_y = float(anchor["humerus_circle_y"])
                    circle_radius = float(anchor["humerus_circle_radius"])
                    arc_residual = 0.35
                    arc_fit_failed = True
                else:
                    arc_fit_failed = False
            dx = edge_x - circle_x
            dy = edge_y - circle_y
            angles = np.degrees(np.arctan2(-dy, dx * lateral_sign))
            z_prior = float(np.exp(-((z / image.shape[2] - 0.515) / 0.075) ** 2))
            for angle_deg in angle_values:
                angle_delta = np.abs(angles - angle_deg)
                support = int(np.sum(angle_delta <= max(5.0, cfg.bone_edge_angle_step_deg * 0.65)))
                if support < 2:
                    continue
                point_index = int(np.argmin(angle_delta))
                edge_point_x = float(edge_x[point_index])
                edge_point_y = float(edge_y[point_index])
                vx = edge_point_x - circle_x
                vy = edge_point_y - circle_y
                norm = float(np.sqrt(vx * vx + vy * vy))
                if norm <= 1e-6:
                    continue
                normal_x = vx / norm
                normal_y = vy / norm
                if normal_x * lateral_sign < -0.10 or normal_y > 0.30:
                    continue
                edge_support_score = float(min(1.0, support / 18.0))
                angle_prior = float(np.exp(-((angle_deg - 65.0) / 32.0) ** 2))
                residual_score = float(np.exp(-((arc_residual) / 0.20) ** 2))
                for offset in offset_values:
                    roi_centers = None
                    roi_half_size = None
                    center = (
                        float(edge_point_x + normal_x * offset),
                        float(edge_point_y + normal_y * offset + cfg.bone_edge_channel_downshift_voxels),
                        float(z),
                    )
                    if cfg.bone_edge_centerline_enable:
                        point_count = max(1, int(cfg.bone_edge_centerline_points))
                        middle = (point_count - 1) / 2.0
                        centers = []
                        for point_index2 in range(point_count):
                            point_angle_deg = angle_deg + (point_index2 - middle) * cfg.bone_edge_centerline_angle_step_deg
                            local_delta = np.abs(angles - point_angle_deg)
                            local_index = int(np.argmin(local_delta))
                            local_edge_x = float(edge_x[local_index])
                            local_edge_y = float(edge_y[local_index])
                            local_vx = local_edge_x - circle_x
                            local_vy = local_edge_y - circle_y
                            local_norm = float(np.sqrt(local_vx * local_vx + local_vy * local_vy))
                            if local_norm <= 1e-6:
                                continue
                            local_normal_x = local_vx / local_norm
                            local_normal_y = local_vy / local_norm
                            centers.append(
                                (
                                    float(local_edge_x + local_normal_x * offset),
                                    float(local_edge_y + local_normal_y * offset + cfg.bone_edge_channel_downshift_voxels),
                                    float(z),
                                )
                            )
                        if len(centers) >= max(1, int(point_count * 0.75)):
                            roi_centers = tuple(centers)
                            roi_half_size = cfg.bone_edge_centerline_half_size
                            center_arr = np.asarray(roi_centers, dtype=float).mean(axis=0)
                            center = (float(center_arr[0]), float(center_arr[1]), float(center_arr[2]))
                            stats = roi_stats_from_centers(image, bone_mask, roi_centers, roi_half_size, cfg)
                        else:
                            stats = None
                    else:
                        stats = roi_stats(image, bone_mask, center, cfg.half_size, cfg)
                    if stats is None:
                        continue
                    (
                        bone_fraction,
                        mean_value,
                        below_bone,
                        above_bone,
                        body_fraction,
                        near_bone_fraction,
                        margin_bone_fraction,
                        bone_distance_mm,
                        distance_band_score,
                    ) = stats
                    if bone_fraction > cfg.bone_edge_max_bone_fraction or body_fraction < 0.94 or not (-30 <= mean_value <= 155):
                        continue
                    band_mean, band_std = soft_tissue_band_stats(
                        image,
                        edge_point_x,
                        edge_point_y,
                        z,
                        normal_x,
                        normal_y,
                        range(1, max(2, int(offset) + 1)),
                    )
                    soft_prior = float(np.exp(-((band_mean - 55.0) / 48.0) ** 2))
                    soft_homogeneity = float(np.exp(-(band_std / 95.0) ** 2))
                    offset_prior = float(
                        np.exp(-((float(offset) - 8.0) / 6.0) ** 2)
                    )
                    contact_score = float(
                        np.exp(-((bone_fraction - cfg.bone_edge_target_bone_fraction) / 0.05) ** 2)
                    )
                    if bone_fraction <= 0.0:
                        contact_score = max(contact_score, min(0.75, margin_bone_fraction * 10.0))
                    anatomical_score = (
                        float(anchor["roof_prior"]) * 0.65
                        + float(anchor["gap_prior"]) * 0.55
                        + z_prior * 0.65
                        + float(anchor.get("head_prior", 0.0)) * 0.50
                        + continuity * 0.55
                        + edge_support_score * 0.85
                        + angle_prior * 0.65
                        + offset_prior * 0.55
                        + residual_score * 0.45
                    )
                    soft_tissue_score = soft_prior * 0.70 + soft_homogeneity * 0.25 + below_bone * 0.08 + above_bone * 0.04
                    deep_bone_excess = max(0.0, bone_fraction - cfg.bone_edge_target_bone_fraction)
                    safety_score = (
                        -deep_bone_excess * 10.0
                        -max(0.0, near_bone_fraction - 0.13) * 0.45
                        -max(0.0, margin_bone_fraction - 0.16) * 0.25
                        + distance_band_score * cfg.bone_dist_band_weight * 0.35
                    )
                    teacher_distance_mm = None
                    if teacher_center is not None:
                        teacher_distance_mm = float(np.sqrt(np.sum(((np.asarray(center) - teacher_center) * np.asarray(cfg.spacing_xyz)) ** 2)))
                    score = (anatomical_score + soft_tissue_score + safety_score) * cfg.bone_edge_weight
                    candidates.append(
                        MultiBonePrediction(
                            center_xyz=center,
                            score=float(score),
                            lateral_sign=lateral_sign,
                            humerus_anchor=humerus_anchor,
                            roof_anchor=roof,
                            roi_stats=stats,
                            global_bone_centroid_x=global_bone_centroid_x,
                            roi_centers_xyz=roi_centers,
                            roi_half_size=roi_half_size,
                            continuity_score=continuity,
                            candidate_source="bone_edge_tendon",
                            anatomical_score=float(anatomical_score),
                            soft_tissue_score=float(soft_tissue_score),
                            safety_score=float(safety_score),
                            bone_distance_band_score=float(distance_band_score),
                            teacher_distance_mm=teacher_distance_mm,
                            radius_normalized_distance=radius_normalized_distance(center, anchor),
                            same_anchor_support_count=support_count,
                            edge_angle_deg=float(angle_deg),
                            edge_point_xyz=(edge_point_x, edge_point_y, float(z)),
                            surface_normal_xy=(float(normal_x), float(normal_y)),
                            surface_offset_voxels=float(offset),
                            surface_distance_mm=float(offset * xy_spacing),
                            cortical_edge_support=edge_support_score,
                            soft_tissue_band_mean=float(band_mean),
                            soft_tissue_band_std=float(band_std),
                            bone_edge_continuity_score=float(continuity),
                            arc_fit_residual=float(arc_residual),
                            bone_edge_tendon_score=float(score),
                            arc_fit_failed=bool(arc_fit_failed),
                        )
                    )

    z_spacing = max(float(cfg.spacing_xyz[2]), 1e-6)
    anchor_models.sort(key=lambda item: float(item["anchor_score"]), reverse=True)
    if anchor_models and cfg.surface_arc_sphere_blend > 0:
        sphere_models = anchor_models[: max(1, int(cfg.surface_arc_sphere_anchor_count))]
        sphere_x = float(np.median([float(item["humerus_circle_x"]) for item in sphere_models]))
        sphere_y = float(np.median([float(item["humerus_circle_y"]) for item in sphere_models]))
        sphere_radius = float(np.median([float(item["humerus_circle_radius"]) for item in sphere_models]))
        blend = float(min(1.0, max(0.0, cfg.surface_arc_sphere_blend)))
        for item in anchor_models:
            item["humerus_circle_x"] = float((1.0 - blend) * float(item["humerus_circle_x"]) + blend * sphere_x)
            item["humerus_circle_y"] = float((1.0 - blend) * float(item["humerus_circle_y"]) + blend * sphere_y)
            item["humerus_circle_radius"] = float((1.0 - blend) * float(item["humerus_circle_radius"]) + blend * sphere_radius)
    current_anchor_models = anchor_models[: max(1, int(cfg.current_anchor_count))]
    branch_anchor_models = anchor_models[: max(1, int(cfg.branch_anchor_count))]
    surface_arc_anchor_models = anchor_models[: max(1, int(cfg.surface_arc_anchor_count))]
    bone_edge_anchor_models = anchor_models[: max(1, int(cfg.bone_edge_anchor_count))]

    for anchor in current_anchor_models:
        anchor_z = int(anchor["z"])
        z_values = [anchor_z]
        add_anchor_candidates(anchor, z_values, "current_multibone", 1.0)
    if cfg.surface_arc_enable:
        for anchor in surface_arc_anchor_models:
            anchor_z = int(anchor["z"])
            z_values = list(
                range(
                    max(0, anchor_z - max(0, int(cfg.surface_arc_z_window))),
                    min(image.shape[2] - 1, anchor_z + max(0, int(cfg.surface_arc_z_window))) + 1,
                )
            )
            add_surface_arc_candidates(anchor, z_values)
    if cfg.bone_edge_enable:
        for anchor in bone_edge_anchor_models:
            anchor_z = int(anchor["z"])
            z_values = list(
                range(
                    max(0, anchor_z - max(0, int(cfg.bone_edge_z_window))),
                    min(image.shape[2] - 1, anchor_z + max(0, int(cfg.bone_edge_z_window))) + 1,
                )
            )
            add_bone_edge_tendon_candidates(anchor, z_values)
    for anchor in branch_anchor_models:
        anchor_z = int(anchor["z"])
        if cfg.low_z_enable:
            low_steps = range(
                int(round(cfg.low_z_step_mm / z_spacing)),
                int(round(cfg.low_z_range_mm / z_spacing)) + 1,
                max(1, int(round(cfg.low_z_step_mm / z_spacing))),
            )
            low_z_values = sorted({anchor_z - step for step in low_steps if anchor_z - step >= 0})
            add_anchor_candidates(anchor, low_z_values, "low_z", cfg.low_z_weight)
        if cfg.teacher_z_refine_enable and cfg.teacher_z_center is not None:
            half_window = int(round(cfg.teacher_z_window_mm / z_spacing))
            step = max(1, int(round(cfg.teacher_z_refine_step_mm / z_spacing)))
            teacher_z = int(round(cfg.teacher_z_center))
            refine_values = list(range(max(0, teacher_z - half_window), min(image.shape[2] - 1, teacher_z + half_window) + 1, step))
            add_anchor_candidates(anchor, refine_values, "teacher_z_refine", 0.96)
        if cfg.teacher_low_z_enable and cfg.teacher_z_center is not None:
            min_shift = max(1, int(round(cfg.teacher_low_z_min_shift_mm / z_spacing)))
            max_shift = max(min_shift, int(round(cfg.teacher_low_z_range_mm / z_spacing)))
            step = max(1, int(round(cfg.teacher_low_z_step_mm / z_spacing)))
            teacher_z = int(round(cfg.teacher_z_center))
            teacher_low_values = sorted({teacher_z - shift for shift in range(min_shift, max_shift + 1, step) if teacher_z - shift >= 0})
            add_anchor_candidates(anchor, teacher_low_values, "teacher_low_z", cfg.teacher_low_z_weight)
        if cfg.contact_z_enable and cfg.teacher_z_center is not None:
            min_shift = max(1, int(round(cfg.contact_z_min_shift_mm / z_spacing)))
            max_shift = max(min_shift, int(round(cfg.contact_z_range_mm / z_spacing)))
            step = max(1, int(round(cfg.contact_z_step_mm / z_spacing)))
            teacher_z = int(round(cfg.teacher_z_center))
            contact_z_values = sorted({teacher_z - shift for shift in range(min_shift, max_shift + 1, step) if teacher_z - shift >= 0})
            add_anchor_candidates(
                anchor,
                contact_z_values,
                "contact_z",
                cfg.contact_z_weight,
                max_bone_fraction=cfg.contact_z_max_bone_fraction,
                bone_penalty_weight=12.0,
                near_bone_penalty_weight=0.45,
                margin_bone_penalty_weight=0.20,
            )
        if cfg.wide_xy_enable:
            dx_values = range(-int(cfg.wide_xy_dx), int(cfg.wide_xy_dx) + 1, max(1, int(cfg.wide_xy_step)))
            dy_values = range(int(cfg.wide_xy_dy_min), int(cfg.wide_xy_dy_max) + 1, max(1, int(cfg.wide_xy_step)))
            add_anchor_candidates(anchor, [anchor_z], "wide_xy", cfg.wide_xy_weight, dx_values=dx_values, dy_values=dy_values)

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates:
        raise ValueError("Could not locate a valid multi-bone supraspinatus sampling ROI")
    if top_k is None:
        return candidates
    balanced: list[MultiBonePrediction] = []
    for source in ("current_multibone", "surface_arc", "bone_edge_tendon", "low_z", "teacher_z_refine", "teacher_low_z", "contact_z", "wide_xy"):
        source_candidates = [item for item in candidates if item.candidate_source == source]
        balanced.extend(source_candidates[:top_k])
    balanced.sort(key=lambda item: item.score, reverse=True)
    return balanced


def locate_multibone_sampling_roi(
    image: np.ndarray,
    config: MultiBoneConfig | None = None,
    bone_mask_override: np.ndarray | None = None,
) -> MultiBonePrediction:
    return locate_multibone_candidates(image, config, top_k=1, bone_mask_override=bone_mask_override)[0]


def prediction_mask_and_bbox(shape: tuple[int, int, int], prediction: MultiBonePrediction, half_size: tuple[int, int, int]) -> tuple[np.ndarray, BBox3D]:
    if prediction.roi_centers_xyz:
        roi_half_size = prediction.roi_half_size or half_size
        mask = np.zeros(shape, dtype=bool)
        for center in prediction.roi_centers_xyz:
            mask |= mask_from_bbox(shape, box_from_center(shape, center, roi_half_size)).astype(bool)
        bbox = bbox_from_mask(mask)
        if bbox is None:
            bbox = box_from_center(shape, prediction.center_xyz, roi_half_size)
        return mask, bbox
    bbox = box_from_center(shape, prediction.center_xyz, half_size)
    return mask_from_bbox(shape, bbox), bbox


def find_first(paths, predicate) -> Path:
    for path in paths:
        if predicate(path):
            return path
    raise FileNotFoundError("No matching file found")


def read_csv_by_case(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def bbox_from_result_row(row: dict[str, str], prefix: str = "pred_box") -> BBox3D:
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


def signed_error_components_mm(pred_center: np.ndarray, doctor_center: np.ndarray, spacing: np.ndarray) -> tuple[float, float, float, float]:
    delta = (pred_center - doctor_center) * spacing
    center_error = float(np.sqrt(np.sum(delta * delta)))
    return center_error, float(delta[0]), float(delta[1]), float(delta[2])


def error_type_label(row: dict[str, object]) -> str:
    err = float(row["center_error_mm"])
    cov = float(row["doctor_roi_coverage"])
    abs_dx = float(row["abs_dx_mm"])
    abs_dy = float(row["abs_dy_mm"])
    abs_dz = float(row["abs_dz_mm"])
    bone_overlap = float(row["pred_bone_overlap"])
    margin = float(row.get("margin_bone_fraction", 0.0))
    if err <= 5.0 and cov >= 0.15:
        return "good_case"
    if abs_dz >= max(abs_dx, abs_dy) * 1.25 and abs_dz >= 4.0:
        return "z_miss"
    if np.sqrt(abs_dx * abs_dx + abs_dy * abs_dy) >= abs_dz * 1.25 and err >= 5.0:
        return "xy_miss"
    if bone_overlap <= 0.002 and margin >= 0.04 and err >= 7.0:
        return "bone_margin_shift"
    return "mixed_miss"


def failure_type_from_topk(top_rows: list[dict[str, object]]) -> str:
    if not top_rows:
        return "generation_failure"
    top1 = top_rows[0]
    best_cov = max(float(row["doctor_roi_coverage"]) for row in top_rows)
    best_err = min(float(row["center_error_mm"]) for row in top_rows)
    top1_cov = float(top1["doctor_roi_coverage"])
    top1_err = float(top1["center_error_mm"])
    if best_cov < 0.05 and best_err > 8.0:
        return "generation_failure"
    if best_cov >= top1_cov + 0.10 or best_err <= top1_err - 2.0:
        return "ranking_failure"
    return str(top1.get("error_type", "mixed_miss"))


def write_csv_union(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate_quality_flags(row: dict[str, object]) -> dict[str, object]:
    anchor_width = float(row["humerus_anchor_x2"]) - float(row["humerus_anchor_x1"]) + 1.0
    anchor_height = float(row["humerus_anchor_y2"]) - float(row["humerus_anchor_y1"]) + 1.0
    detached = (
        float(row.get("bone_overlap") or row.get("pred_bone_overlap") or 0.0) <= 0.001
        and float(row.get("near_bone_fraction") or 0.0) <= 0.005
        and float(row.get("margin_bone_fraction") or 0.0) <= 0.005
    )
    suspicious_narrow = anchor_width <= 55.0 or anchor_height <= 34.0
    radius_dist = row.get("radius_normalized_distance")
    try:
        radius_dist_float = float(radius_dist)
    except (TypeError, ValueError):
        radius_dist_float = 0.0
    possible_wrong = bool(
        (detached and suspicious_narrow)
        or radius_dist_float < 0.72
        or radius_dist_float > 1.55
        or int(float(row.get("same_anchor_support_count") or 0)) <= 0
    )
    return {
        "humerus_anchor_width": round(anchor_width, 3),
        "humerus_anchor_height": round(anchor_height, 3),
        "z_continuity_score": row.get("continuity_score", ""),
        "detached_from_bone": bool(detached),
        "suspicious_narrow_anchor": bool(suspicious_narrow),
        "possible_wrong_bone": bool(possible_wrong),
    }


def same_humerus_anchor(row: dict[str, object], reference: dict[str, object]) -> bool:
    dx = abs(float(row["humerus_anchor_cx"]) - float(reference["humerus_anchor_cx"]))
    dy = abs(float(row["humerus_anchor_cy"]) - float(reference["humerus_anchor_cy"]))
    ref_width = max(1.0, float(reference["humerus_anchor_x2"]) - float(reference["humerus_anchor_x1"]) + 1.0)
    ref_height = max(1.0, float(reference["humerus_anchor_y2"]) - float(reference["humerus_anchor_y1"]) + 1.0)
    return dx <= max(55.0, ref_width * 0.75) and dy <= max(32.0, ref_height * 0.60)


def select_generalized_candidate(model_rows: list[dict[str, object]], cfg: MultiBoneConfig) -> tuple[dict[str, object], str]:
    sorted_rows = sorted(model_rows, key=lambda row: float(row["total_score"]), reverse=True)
    current_rows = [row for row in sorted_rows if row["candidate_source"] == "current_multibone"]
    primary = current_rows[0] if current_rows else sorted_rows[0]
    primary_gap = float(primary["center_y_minus_humerus_top"])
    primary_anchor_width = float(primary["humerus_anchor_x2"]) - float(primary["humerus_anchor_x1"]) + 1.0
    same_anchor_rows = [row for row in sorted_rows if same_humerus_anchor(row, primary)]

    surface_rows = [
        row
        for row in same_anchor_rows
        if row["candidate_source"] == "surface_arc"
        and -2.0 <= float(row["center_y_minus_humerus_top"]) <= 32.0
        and 0.010 <= float(row["bone_overlap"]) <= min(0.085, cfg.surface_arc_max_bone_fraction)
        and float(row.get("near_bone_fraction") or 0.0) <= 0.11
        and float(row.get("margin_bone_fraction") or 0.0) <= 0.13
        and float(row.get("body_inside_fraction") or 0.0) >= 0.94
        and float(row.get("z_continuity_score") or row.get("continuity_score") or 0.0) >= 0.75
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
        for row in sorted_rows
        if row["candidate_source"] == "surface_arc"
        and -2.0 <= float(row["center_y_minus_humerus_top"]) <= 24.0
        and 0.010 <= float(row["bone_overlap"]) <= min(0.085, cfg.surface_arc_max_bone_fraction)
        and float(row.get("near_bone_fraction") or 0.0) <= 0.105
        and float(row.get("margin_bone_fraction") or 0.0) <= 0.115
        and float(row.get("body_inside_fraction") or 0.0) >= 0.94
        and float(row.get("z_continuity_score") or row.get("continuity_score") or 0.0) >= 0.75
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


def process_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    config: MultiBoneConfig | None = None,
    teacher_csv: str | Path | None = None,
    case_names: set[str] | None = None,
    selection_policy: str = "legacy",
    export_candidates: bool = False,
    candidate_preview_topk: int = 5,
    bone_mask_dir: str | Path | None = None,
    bone_mask_filename: str = "shoulder_bones_combined.nii.gz",
    allow_threshold_bone_fallback: bool = False,
    min_external_bone_voxels: int = 10000,
) -> None:
    cfg = config or MultiBoneConfig()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    previews_dir = output_dir / "previews"
    reports_dir = output_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    teacher_rows = read_csv_by_case(Path(teacher_csv) if teacher_csv is not None else None)
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for case_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if case_names is not None and case_dir.name not in case_names:
            continue
        ct_dir = case_dir / "CT"
        ct_60_path = find_first(ct_dir.iterdir(), lambda p: p.is_file() and "60" in p.name.lower() and ".nii" in p.name.lower())
        roi_path = find_first(ct_dir.iterdir(), lambda p: p.is_file() and "roi" in p.name.lower() and ".nii" in p.name.lower())
        image_volume: NiftiImage = load_nifti(ct_60_path)
        image = image_volume.data.astype(np.float32)
        bone_mask_override = None
        bone_mask_source = "threshold"
        external_bone_voxels = ""
        if bone_mask_dir is not None:
            external_bone_path = Path(bone_mask_dir) / case_dir.name / bone_mask_filename
            if external_bone_path.exists():
                bone_mask_override = load_mask_compatible(external_bone_path, image.shape)
                external_bone_voxels = int(np.asarray(bone_mask_override, dtype=bool).sum())
                if external_bone_voxels < min_external_bone_voxels:
                    if allow_threshold_bone_fallback:
                        print(
                            f"{case_dir.name}: external bone mask has {external_bone_voxels} voxels "
                            f"(< {min_external_bone_voxels}); falling back to HU threshold."
                        )
                        bone_mask_override = None
                        bone_mask_source = f"threshold_fallback_invalid_external:{external_bone_path}"
                    else:
                        raise ValueError(
                            f"External bone mask for {case_dir.name} is too small "
                            f"({external_bone_voxels} voxels < {min_external_bone_voxels}): {external_bone_path}. "
                            "Rerun TotalSegmentator for this case or pass --allow-threshold-bone-fallback."
                        )
                else:
                    bone_mask_source = str(external_bone_path)
            elif not allow_threshold_bone_fallback:
                raise FileNotFoundError(f"External bone mask not found for {case_dir.name}: {external_bone_path}")
            else:
                bone_mask_source = f"threshold_fallback_missing_external:{external_bone_path}"
        doctor_roi = load_nifti(roi_path).data != 0
        doctor_box = bbox_from_mask(doctor_roi)
        if doctor_box is None:
            raise ValueError(f"Empty ROI for {case_dir.name}")
        spacing = np.asarray(image_volume.spacing[:3], dtype=float)
        cfg.spacing_xyz = tuple(float(v) for v in spacing)
        teacher_row = teacher_rows.get(case_dir.name)
        if teacher_row is not None:
            cfg.teacher_z_center = float(teacher_row["pred_center_z"])
            cfg.teacher_center_xyz = (
                float(teacher_row["pred_center_x"]),
                float(teacher_row["pred_center_y"]),
                float(teacher_row["pred_center_z"]),
            )
        else:
            cfg.teacher_z_center = None
            cfg.teacher_center_xyz = None

        try:
            candidates = locate_multibone_candidates(image, cfg, top_k=max(cfg.top_k * 4, 20), bone_mask_override=bone_mask_override)
        except ValueError as exc:
            if bone_mask_override is not None and allow_threshold_bone_fallback:
                print(f"{case_dir.name}: candidate generation failed with external bone mask ({exc}); retrying HU threshold.")
                bone_mask_override = None
                bone_mask_source = f"threshold_fallback_generation_failed:{bone_mask_source}"
                candidates = locate_multibone_candidates(image, cfg, top_k=max(cfg.top_k * 4, 20), bone_mask_override=None)
            else:
                raise ValueError(f"{case_dir.name}: {exc}") from exc
        bone_mask = bone_mask_override if bone_mask_override is not None else image > cfg.bone_threshold_hu
        doctor_center = np.argwhere(doctor_roi).mean(axis=0)

        evaluated_rows: list[dict[str, object]] = []
        evaluated_masks: dict[str, np.ndarray] = {}
        evaluated_boxes: dict[str, BBox3D] = {}

        for idx, candidate in enumerate(candidates, start=1):
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
            candidate_center = np.asarray(candidate.center_xyz)
            candidate_error_mm, dx_mm, dy_mm, dz_mm = signed_error_components_mm(candidate_center, doctor_center, spacing)
            row = {
                "case": case_dir.name,
                "bone_mask_source": bone_mask_source,
                "external_bone_voxels": external_bone_voxels,
                "candidate_id": f"{case_dir.name}_{idx:03d}",
                "candidate_source": candidate.candidate_source,
                "rank": idx,
                "total_score": round(candidate.score, 4),
                "anatomical_score": round(candidate.anatomical_score, 4),
                "soft_tissue_score": round(candidate.soft_tissue_score, 4),
                "safety_score": round(candidate.safety_score, 4),
                "continuity_score": round(float(candidate.continuity_score), 4),
                "same_anchor_support_count": int(candidate.same_anchor_support_count),
                "near_bone_fraction": round(float(candidate_near_bone_fraction), 4),
                "margin_bone_fraction": round(float(candidate_margin_bone_fraction), 4),
                "bone_distance_mm": round(float(candidate_bone_distance_mm), 3),
                "bone_distance_band_score": round(float(candidate_band_score), 4),
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
                "pred_center_x": round(float(candidate_center[0]), 2),
                "pred_center_y": round(float(candidate_center[1]), 2),
                "pred_center_z": round(float(candidate_center[2]), 2),
                "center_y_minus_humerus_top": round(float(candidate_center[1] - candidate.humerus_anchor.y1), 3),
                "humerus_anchor_cx": round(float(candidate.humerus_anchor.cx), 3),
                "humerus_anchor_cy": round(float(candidate.humerus_anchor.cy), 3),
                "pred_center_phys_x": round(float(candidate_center[0] * spacing[0]), 3),
                "pred_center_phys_y": round(float(candidate_center[1] * spacing[1]), 3),
                "pred_center_phys_z": round(float(candidate_center[2] * spacing[2]), 3),
                "pred_box_x1": candidate_box.min[0],
                "pred_box_y1": candidate_box.min[1],
                "pred_box_z1": candidate_box.min[2],
                "pred_box_x2": candidate_box.max[0],
                "pred_box_y2": candidate_box.max[1],
                "pred_box_z2": candidate_box.max[2],
                "bbox_size_phys_x": round(float((candidate_box.max[0] - candidate_box.min[0] + 1) * spacing[0]), 3),
                "bbox_size_phys_y": round(float((candidate_box.max[1] - candidate_box.min[1] + 1) * spacing[1]), 3),
                "bbox_size_phys_z": round(float((candidate_box.max[2] - candidate_box.min[2] + 1) * spacing[2]), 3),
                "center_error_mm": round(candidate_error_mm, 2),
                "dx_mm": round(dx_mm, 3),
                "dy_mm": round(dy_mm, 3),
                "dz_mm": round(dz_mm, 3),
                "abs_dx_mm": round(abs(dx_mm), 3),
                "abs_dy_mm": round(abs(dy_mm), 3),
                "abs_dz_mm": round(abs(dz_mm), 3),
                "doctor_roi_coverage": round(float((doctor_roi & candidate_mask).sum() / max(1, doctor_roi.sum())), 4),
                "bbox_iou": round(bbox_iou(candidate_box, doctor_box), 4),
                "pred_box_doctor_bbox_iou": round(bbox_iou(candidate_box, doctor_box), 4),
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
            row["error_type"] = error_type_label(
                {
                    **row,
                    "pred_bone_overlap": row["bone_overlap"],
                }
            )
            evaluated_rows.append(row)
            evaluated_masks[str(row["candidate_id"])] = candidate_mask
            evaluated_boxes[str(row["candidate_id"])] = candidate_box

        if teacher_row is not None:
            teacher_box = bbox_from_result_row(teacher_row)
            teacher_mask = mask_from_bbox(image.shape, teacher_box)
            teacher_center = np.asarray(
                [float(teacher_row["pred_center_x"]), float(teacher_row["pred_center_y"]), float(teacher_row["pred_center_z"])],
                dtype=float,
            )
            teacher_error, tdx, tdy, tdz = signed_error_components_mm(teacher_center, doctor_center, spacing)
            teacher_candidate = {
                "case": case_dir.name,
                "candidate_id": f"{case_dir.name}_teacher",
                "candidate_source": "teacher_baseline",
                "rank": 999,
                "total_score": round(float(teacher_row.get("score", 0.0) or 0.0), 4),
                "anatomical_score": "",
                "soft_tissue_score": "",
                "safety_score": "",
                "continuity_score": "",
                "near_bone_fraction": "",
                "margin_bone_fraction": "",
                "bone_distance_mm": "",
                "bone_distance_band_score": "",
                "bone_overlap": round(float((teacher_mask & bone_mask).sum() / max(1, teacher_mask.sum())), 4),
                "body_inside_fraction": "",
                "teacher_distance_mm": 0.0,
                "arc_angle_deg": "",
                "surface_offset_voxels": "",
                "radius_normalized_distance": "",
                "edge_angle_deg": "",
                "edge_point_x": "",
                "edge_point_y": "",
                "edge_point_z": "",
                "surface_normal_x": "",
                "surface_normal_y": "",
                "surface_distance_mm": "",
                "cortical_edge_support": "",
                "soft_tissue_band_mean": "",
                "soft_tissue_band_std": "",
                "bone_edge_continuity_score": "",
                "arc_fit_residual": "",
                "bone_edge_tendon_score": "",
                "arc_fit_failed": "",
                "pred_center_x": round(float(teacher_center[0]), 2),
                "pred_center_y": round(float(teacher_center[1]), 2),
                "pred_center_z": round(float(teacher_center[2]), 2),
                "pred_center_phys_x": round(float(teacher_center[0] * spacing[0]), 3),
                "pred_center_phys_y": round(float(teacher_center[1] * spacing[1]), 3),
                "pred_center_phys_z": round(float(teacher_center[2] * spacing[2]), 3),
                "pred_box_x1": teacher_box.min[0],
                "pred_box_y1": teacher_box.min[1],
                "pred_box_z1": teacher_box.min[2],
                "pred_box_x2": teacher_box.max[0],
                "pred_box_y2": teacher_box.max[1],
                "pred_box_z2": teacher_box.max[2],
                "bbox_size_phys_x": round(float((teacher_box.max[0] - teacher_box.min[0] + 1) * spacing[0]), 3),
                "bbox_size_phys_y": round(float((teacher_box.max[1] - teacher_box.min[1] + 1) * spacing[1]), 3),
                "bbox_size_phys_z": round(float((teacher_box.max[2] - teacher_box.min[2] + 1) * spacing[2]), 3),
                "center_error_mm": round(teacher_error, 2),
                "dx_mm": round(tdx, 3),
                "dy_mm": round(tdy, 3),
                "dz_mm": round(tdz, 3),
                "abs_dx_mm": round(abs(tdx), 3),
                "abs_dy_mm": round(abs(tdy), 3),
                "abs_dz_mm": round(abs(tdz), 3),
                "doctor_roi_coverage": round(float((doctor_roi & teacher_mask).sum() / max(1, doctor_roi.sum())), 4),
                "bbox_iou": round(bbox_iou(teacher_box, doctor_box), 4),
                "pred_box_doctor_bbox_iou": round(bbox_iou(teacher_box, doctor_box), 4),
                "pred_soft_mean_60kev": round(float(image[teacher_mask > 0].mean()), 2),
                "bone_fraction": "",
            }
            teacher_candidate["error_type"] = error_type_label({**teacher_candidate, "pred_bone_overlap": teacher_candidate["bone_overlap"], "margin_bone_fraction": 0.0})
            evaluated_rows.append(teacher_candidate)
            evaluated_masks[str(teacher_candidate["candidate_id"])] = teacher_mask
            evaluated_boxes[str(teacher_candidate["candidate_id"])] = teacher_box

        model_rows = [row for row in evaluated_rows if row["candidate_source"] != "teacher_baseline"]
        model_rows.sort(key=lambda row: float(row["total_score"]), reverse=True)
        current_rows = [row for row in model_rows if row["candidate_source"] == "current_multibone"]
        current_rows.sort(key=lambda row: float(row["total_score"]), reverse=True)
        surface_arc_rows_all = [row for row in model_rows if row["candidate_source"] == "surface_arc"]
        bone_edge_rows_all = [row for row in model_rows if row["candidate_source"] == "bone_edge_tendon"]
        low_z_rows_all = [row for row in model_rows if row["candidate_source"] == "low_z"]
        teacher_refine_rows_all = [row for row in model_rows if row["candidate_source"] == "teacher_z_refine"]
        teacher_low_z_rows_all = [row for row in model_rows if row["candidate_source"] == "teacher_low_z"]
        contact_z_rows_all = [row for row in model_rows if row["candidate_source"] == "contact_z"]
        wide_xy_rows_all = [row for row in model_rows if row["candidate_source"] == "wide_xy"]
        source_top_rows = (
            current_rows[: cfg.top_k]
            + surface_arc_rows_all[: cfg.top_k]
            + bone_edge_rows_all[: cfg.top_k]
            + low_z_rows_all[: cfg.top_k]
            + teacher_refine_rows_all[: cfg.top_k]
            + teacher_low_z_rows_all[: cfg.top_k]
            + contact_z_rows_all[: cfg.top_k]
            + wide_xy_rows_all[: cfg.top_k]
        )
        source_top_rows.sort(key=lambda row: float(row["total_score"]), reverse=True)
        for rank, row in enumerate(source_top_rows, start=1):
            row["global_rank"] = rank
            row["source_rank"] = 1 + sum(
                1
                for other in source_top_rows
                if other["candidate_source"] == row["candidate_source"] and float(other["total_score"]) > float(row["total_score"])
            )
            candidate_rows.append(row)
        if export_candidates:
            top_preview_rows = source_top_rows[: max(1, int(candidate_preview_topk))]
            save_candidate_sheet_pil(
                image,
                previews_dir / f"{case_dir.name}_candidate_top{len(top_preview_rows)}.png",
                top_preview_rows,
                evaluated_masks,
                doctor_roi=doctor_roi,
            )

        top1 = current_rows[0] if current_rows else source_top_rows[0]
        safe_rows = [
            row
            for row in source_top_rows
            if float(row["bone_overlap"]) <= 0.006 and float(row.get("near_bone_fraction") or 0.0) <= 0.08
        ]
        selected = top1
        decision_reason = "choose_multibone_high_confidence"
        if float(top1["bone_overlap"]) > 0.011 or float(top1.get("near_bone_fraction") or 0.0) > 0.12:
            if safe_rows and float(safe_rows[0]["total_score"]) >= float(top1["total_score"]) - 0.45:
                selected = safe_rows[0]
                decision_reason = "choose_topk_lower_bone_risk"
            elif teacher_row is not None:
                selected = next(row for row in evaluated_rows if row["candidate_source"] == "teacher_baseline")
                decision_reason = "fallback_teacher_high_bone_risk"
        elif teacher_row is not None and cfg.teacher_center_xyz is not None:
            refine_rows = [row for row in model_rows if row["candidate_source"] == "teacher_z_refine"]
            top1_center = np.asarray([float(top1["pred_center_x"]), float(top1["pred_center_y"]), float(top1["pred_center_z"])])
            teacher_center = np.asarray(cfg.teacher_center_xyz)
            z_gap = abs(float((top1_center[2] - teacher_center[2]) * spacing[2]))
            if refine_rows and float(top1["total_score"]) < 3.2 and z_gap >= 7.0 and float(top1.get("continuity_score") or 0.0) < 0.45:
                selected = refine_rows[0]
                decision_reason = "choose_teacher_z_refine_large_z_gap"
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
            and 0.045 <= float(row.get("near_bone_fraction") or 0.0) <= 0.12
            and -4.0 <= float(row["center_y_minus_humerus_top"]) <= 9.5
            and float(row.get("body_inside_fraction") or 0.0) >= 0.94
            and abs(float(row["pred_center_x"]) - top1_x) <= 18.5
            and 12.0 <= float(row["pred_center_y"]) - top1_y <= 19.5
            and abs(float(row["pred_center_z"]) - top1_z) <= 4.0
            and float(row.get("z_continuity_score") or row.get("continuity_score") or 0.0) >= 0.75
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
        if (
            cfg.surface_arc_select_enable
            and qualified_surface_arc_rows
        ):
            selected = qualified_surface_arc_rows[0]
            decision_reason = "choose_surface_arc_humeral_head_candidate"
        teacher_low_z_rows = [row for row in model_rows if row["candidate_source"] == "teacher_low_z"]
        if (
            cfg.teacher_low_z_select_enable
            and teacher_low_z_rows
            and float(teacher_low_z_rows[0]["total_score"]) >= float(top1["total_score"]) - 0.35
            and float(teacher_low_z_rows[0]["bone_overlap"]) <= 0.006
            and float(teacher_low_z_rows[0].get("near_bone_fraction") or 0.0) <= 0.08
            and float(top1.get("continuity_score") or 0.0) < 0.55
            and float(top1["pred_center_z"]) - float(teacher_low_z_rows[0]["pred_center_z"]) <= 8.0
        ):
            selected = teacher_low_z_rows[0]
            decision_reason = "choose_teacher_low_z_candidate_generation_fix"
        contact_z_rows = [row for row in model_rows if row["candidate_source"] == "contact_z"]
        if (
            cfg.contact_z_select_enable
            and contact_z_rows
            and float(contact_z_rows[0]["total_score"]) >= float(top1["total_score"]) - 1.40
            and float(contact_z_rows[0]["bone_overlap"]) >= 0.04
            and float(contact_z_rows[0]["bone_overlap"]) <= cfg.contact_z_max_bone_fraction
            and float(contact_z_rows[0].get("body_inside_fraction") or 0.0) >= 0.94
            and float(top1["pred_center_z"]) - float(contact_z_rows[0]["pred_center_z"]) >= 5.0
        ):
            selected = contact_z_rows[0]
            decision_reason = "choose_contact_z_insertion_candidate"
        wide_xy_rows = [row for row in model_rows if row["candidate_source"] == "wide_xy"]
        if (
            cfg.wide_xy_select_enable
            and wide_xy_rows
            and float(wide_xy_rows[0]["total_score"]) >= float(top1["total_score"]) - 0.30
            and float(wide_xy_rows[0]["bone_overlap"]) <= 0.006
            and float(wide_xy_rows[0].get("near_bone_fraction") or 0.0) <= 0.08
            and (float(top1["total_score"]) < 3.65 or float(top1.get("continuity_score") or 0.0) < 0.45)
        ):
            selected = wide_xy_rows[0]
            decision_reason = "choose_wide_xy_candidate_generation_fix"
        if teacher_row is not None and not safe_rows and float(top1["bone_overlap"]) > 0.010:
            selected = next(row for row in evaluated_rows if row["candidate_source"] == "teacher_baseline")
            decision_reason = "fallback_teacher_all_multibone_high_risk"
        if (
            teacher_row is not None
            and selected["candidate_source"] == "current_multibone"
            and float(selected["total_score"]) < 3.7
            and float(selected.get("teacher_distance_mm") or 0.0) > 7.0
            and float(selected.get("margin_bone_fraction") or 0.0) > 0.035
        ):
            selected = next(row for row in evaluated_rows if row["candidate_source"] == "teacher_baseline")
            decision_reason = "fallback_teacher_possible_bone_margin_shift"
        if (
            teacher_row is not None
            and selected["candidate_source"] == "current_multibone"
            and float(selected["total_score"]) < 3.20
            and str(selected.get("possible_wrong_bone", "")).lower() == "true"
            and teacher_refine_rows_all
            and float(teacher_refine_rows_all[0]["total_score"]) >= float(selected["total_score"]) + 0.70
        ):
            selected = teacher_refine_rows_all[0]
            decision_reason = "fallback_teacher_refine_possible_wrong_bone"

        if selection_policy == "generalized":
            selected, decision_reason = select_generalized_candidate(model_rows, cfg)

        selected_id = str(selected["candidate_id"])
        selected_mask = evaluated_masks[selected_id]
        selected_box = evaluated_boxes[selected_id]
        save_nifti_like(output_dir / f"{case_dir.name}_multibone_roi.nii.gz", selected_mask.astype(np.uint8), reference=image_volume)
        save_nifti_like(results_dir / f"{case_dir.name}_final_roi.nii.gz", selected_mask.astype(np.uint8), reference=image_volume)
        save_overlay_montage_pil(
            image,
            previews_dir / f"{case_dir.name}_top1_preview.png",
            masks=[(selected_mask, (64, 220, 255)), (doctor_roi, (255, 64, 64))],
            center=int(round(float(selected["pred_center_z"]))),
        )
        save_overlay_montage_pil(
            image,
            output_dir / f"{case_dir.name}_multibone_preview.png",
            masks=[(selected_mask, (64, 220, 255)), (doctor_roi, (255, 64, 64))],
            center=int(round(float(selected["pred_center_z"]))),
        )

        row = {
            **selected,
            "ct_60_file": ct_60_path.name,
            "roi_file": roi_path.name,
            "method": "generalized_surface_arc_rescue" if selection_policy == "generalized" else "next_round_multifactor_fusion",
            "selection_policy": selection_policy,
            "selected_candidate_id": selected_id,
            "selected_method": selected["candidate_source"],
            "decision_reason": decision_reason,
            "pred_center_to_doctor_3d_bbox_mm": round(center_to_box_gap_mm(np.asarray([float(selected["pred_center_x"]), float(selected["pred_center_y"]), float(selected["pred_center_z"])]), doctor_box, spacing), 2),
            "pred_box_intersects_doctor_3d_bbox": bool(np.any(selected_mask & doctor_roi)),
            "pred_box_to_doctor_3d_bbox_mm": round(box_to_box_gap_mm(selected_box, doctor_box, spacing), 2),
            "pred_bone_overlap": selected["bone_overlap"],
            "score": selected["total_score"],
            "doctor_center_x": round(float(doctor_center[0]), 2),
            "doctor_center_y": round(float(doctor_center[1]), 2),
            "doctor_center_z": round(float(doctor_center[2]), 2),
            "doctor_bbox_x1": doctor_box.min[0],
            "doctor_bbox_y1": doctor_box.min[1],
            "doctor_bbox_z1": doctor_box.min[2],
            "doctor_bbox_x2": doctor_box.max[0],
            "doctor_bbox_y2": doctor_box.max[1],
            "doctor_bbox_z2": doctor_box.max[2],
        }
        rows.append(row)
        final_rows.append(row)

        ranked_for_analysis = [selected] + [row for row in source_top_rows if row["candidate_id"] != selected_id]
        top3 = ranked_for_analysis[:3]
        top5 = ranked_for_analysis[:5]
        failure_type = failure_type_from_topk(top5)
        failure_rows.append(
            {
                "case": case_dir.name,
                "top1_center_error_mm": top1["center_error_mm"],
                "top1_coverage": top1["doctor_roi_coverage"],
                "top1_iou": top1["bbox_iou"],
                "top1_bone_overlap": top1["bone_overlap"],
                "top3_best_coverage": round(max(float(row["doctor_roi_coverage"]) for row in top3), 4),
                "top5_best_coverage": round(max(float(row["doctor_roi_coverage"]) for row in top5), 4),
                "top3_min_center_error": round(min(float(row["center_error_mm"]) for row in top3), 2),
                "top5_min_center_error": round(min(float(row["center_error_mm"]) for row in top5), 2),
                "failure_type": failure_type,
                "final_selected_source": selected["candidate_source"],
                "decision_reason": decision_reason,
            }
        )

    top1_rows = [row for row in candidate_rows if int(row["rank"]) == 1]
    write_csv_union(output_dir / "multibone_locator_results.csv", rows)
    write_csv_union(output_dir / "multibone_topk_candidates.csv", candidate_rows)
    write_csv_union(results_dir / "per_case_final.csv", final_rows)
    write_csv_union(results_dir / "per_case_top1.csv", top1_rows)
    write_csv_union(results_dir / "per_case_topk.csv", candidate_rows)
    write_csv_union(results_dir / "failure_analysis.csv", failure_rows)

    coverage = np.asarray([float(row["doctor_roi_coverage"]) for row in rows])
    errors = np.asarray([float(row["center_error_mm"]) for row in rows])
    ious = np.asarray([float(row["pred_box_doctor_bbox_iou"]) for row in rows])
    bone_overlap = np.asarray([float(row["pred_bone_overlap"]) for row in rows])
    abs_dx = np.asarray([float(row["abs_dx_mm"]) for row in rows])
    abs_dy = np.asarray([float(row["abs_dy_mm"]) for row in rows])
    abs_dz = np.asarray([float(row["abs_dz_mm"]) for row in rows])
    top3_best_cov = np.asarray([float(row["top3_best_coverage"]) for row in failure_rows])
    top5_best_cov = np.asarray([float(row["top5_best_coverage"]) for row in failure_rows])
    summary_rows = [
        {
            "method": "generalized_surface_arc_rescue" if selection_policy == "generalized" else "next_round_multifactor_fusion",
            "selection_policy": selection_policy,
            "cases": len(rows),
            "mean_center_error_mm": round(float(errors.mean()), 3),
            "median_center_error_mm": round(float(np.median(errors)), 3),
            "worst_center_error_mm": round(float(errors.max()), 3),
            "mean_bbox_iou": round(float(ious.mean()), 4),
            "mean_doctor_roi_coverage": round(float(coverage.mean()), 4),
            "mean_pred_bone_overlap": round(float(bone_overlap.mean()), 4),
            "mean_abs_dx_mm": round(float(abs_dx.mean()), 3),
            "mean_abs_dy_mm": round(float(abs_dy.mean()), 3),
            "mean_abs_dz_mm": round(float(abs_dz.mean()), 3),
            "median_abs_dx_mm": round(float(np.median(abs_dx)), 3),
            "median_abs_dy_mm": round(float(np.median(abs_dy)), 3),
            "median_abs_dz_mm": round(float(np.median(abs_dz)), 3),
            "top3_best_coverage_mean": round(float(top3_best_cov.mean()), 4),
            "top5_best_coverage_mean": round(float(top5_best_cov.mean()), 4),
            "generation_failure_count": sum(1 for row in failure_rows if row["failure_type"] == "generation_failure"),
            "ranking_failure_count": sum(1 for row in failure_rows if row["failure_type"] == "ranking_failure"),
        }
    ]
    write_csv_union(results_dir / "summary_metrics.csv", summary_rows)

    worst_dz = sorted(rows, key=lambda row: float(row["abs_dz_mm"]), reverse=True)[:3]
    worst_error = sorted(rows, key=lambda row: float(row["center_error_mm"]), reverse=True)[:3]
    summary_text = "\n".join(
        [
            "Next-round multi-bone CT supraspinatus sampling ROI locator",
            f"cases: {len(rows)}",
            f"mean_center_error_mm: {errors.mean():.3f}",
            f"median_center_error_mm: {np.median(errors):.3f}",
            f"worst_center_error_mm: {errors.max():.3f}",
            f"mean_bbox_iou: {ious.mean():.4f}",
            f"mean_doctor_roi_coverage: {coverage.mean():.4f}",
            f"mean_pred_bone_overlap: {bone_overlap.mean():.4f}",
            f"mean_abs_dx_mm: {abs_dx.mean():.3f}",
            f"mean_abs_dy_mm: {abs_dy.mean():.3f}",
            f"mean_abs_dz_mm: {abs_dz.mean():.3f}",
            f"median_abs_dx_mm: {np.median(abs_dx):.3f}",
            f"median_abs_dy_mm: {np.median(abs_dy):.3f}",
            f"median_abs_dz_mm: {np.median(abs_dz):.3f}",
            "worst_abs_dz_cases: " + ", ".join(f"{row['case']}={row['abs_dz_mm']}" for row in worst_dz),
            "worst_center_error_cases: " + ", ".join(f"{row['case']}={row['center_error_mm']}" for row in worst_error),
            f"generation_failure_count: {summary_rows[0]['generation_failure_count']}",
            f"ranking_failure_count: {summary_rows[0]['ranking_failure_count']}",
            "anchors: proximal humerus upper-lateral anchor + acromion/scapular roof component",
            f"top_k: {cfg.top_k}",
            f"current_anchor_count: {cfg.current_anchor_count}",
            f"low_z_enable: {cfg.low_z_enable}",
            f"branch_anchor_count: {cfg.branch_anchor_count}",
            f"teacher_low_z_enable: {cfg.teacher_low_z_enable}",
            f"teacher_low_z_select_enable: {cfg.teacher_low_z_select_enable}",
            f"contact_z_enable: {cfg.contact_z_enable}",
            f"contact_z_select_enable: {cfg.contact_z_select_enable}",
            f"teacher_z_refine_enable: {cfg.teacher_z_refine_enable}",
            f"wide_xy_enable: {cfg.wide_xy_enable}",
            f"wide_xy_select_enable: {cfg.wide_xy_select_enable}",
            f"surface_arc_enable: {cfg.surface_arc_enable}",
            f"surface_arc_select_enable: {cfg.surface_arc_select_enable}",
            f"surface_arc_anchor_count: {cfg.surface_arc_anchor_count}",
            f"bone_edge_enable: {cfg.bone_edge_enable}",
            f"bone_edge_anchor_count: {cfg.bone_edge_anchor_count}",
            f"bone_edge_centerline_enable: {cfg.bone_edge_centerline_enable}",
            f"bone_edge_centerline_points: {cfg.bone_edge_centerline_points}",
            f"bone_edge_channel_downshift_voxels: {cfg.bone_edge_channel_downshift_voxels}",
            f"bone_margin_voxels: {cfg.bone_margin_voxels}",
            f"continuity_window: {cfg.continuity_window}",
            f"continuity_xy_tolerance: {cfg.continuity_xy_tolerance}",
            f"bone_dist_good_min_mm: {cfg.bone_dist_good_min_mm}",
            f"bone_dist_good_max_mm: {cfg.bone_dist_good_max_mm}",
            f"selection_policy: {selection_policy}",
        ]
    )
    (output_dir / "multibone_locator_summary.txt").write_text(summary_text, encoding="utf-8")
    (reports_dir / "summary.txt").write_text(summary_text, encoding="utf-8")


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    finite = slice_2d[np.isfinite(slice_2d)]
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((slice_2d - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def save_overlay_montage_pil(
    image: np.ndarray,
    out_path: Path,
    masks: list[tuple[np.ndarray, tuple[int, int, int]]],
    center: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = [max(0, center - 2), center, min(image.shape[2] - 1, center + 2)]
    tiles = []
    for z in indices:
        base = Image.fromarray(normalize_slice(image[:, :, z].T)).convert("RGB").resize((512, 512))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        sx = base.size[0] / image.shape[0]
        sy = base.size[1] / image.shape[1]
        for mask, color in masks:
            points = np.argwhere(mask[:, :, z] > 0)
            if len(points) == 0:
                continue
            for x, y in points[:: max(1, len(points) // 4000)]:
                draw.rectangle(
                    [int(x * sx), int(y * sy), int((x + 1) * sx) + 1, int((y + 1) * sy) + 1],
                    fill=(*color, 115),
                )
        composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        ImageDraw.Draw(composed).text((8, 8), f"slice {z} cyan=multi-bone red=doctor", fill=(255, 255, 0))
        tiles.append(composed)
    sheet = Image.new("RGB", (512 * len(tiles), 512), "black")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, (idx * 512, 0))
    sheet.save(out_path)


def _draw_bbox_2d(draw: ImageDraw.ImageDraw, row: dict[str, object], scale_x: float, scale_y: float, color: tuple[int, int, int], width: int = 3) -> None:
    x1 = int(float(row["pred_box_x1"]) * scale_x)
    y1 = int(float(row["pred_box_y1"]) * scale_y)
    x2 = int((float(row["pred_box_x2"]) + 1.0) * scale_x)
    y2 = int((float(row["pred_box_y2"]) + 1.0) * scale_y)
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def save_candidate_sheet_pil(
    image: np.ndarray,
    out_path: Path,
    candidate_rows: list[dict[str, object]],
    candidate_masks: dict[str, np.ndarray],
    doctor_roi: np.ndarray | None = None,
    tile_size: int = 384,
) -> None:
    """Write a compact top-k candidate preview sheet for manual review."""
    if not candidate_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for idx, row in enumerate(candidate_rows, start=1):
        z = int(round(float(row["pred_center_z"])))
        z = max(0, min(image.shape[2] - 1, z))
        base = Image.fromarray(normalize_slice(image[:, :, z].T)).convert("RGB").resize((tile_size, tile_size))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        sx = base.size[0] / image.shape[0]
        sy = base.size[1] / image.shape[1]

        candidate_id = str(row["candidate_id"])
        mask = candidate_masks.get(candidate_id)
        if mask is not None and z < mask.shape[2]:
            points = np.argwhere(mask[:, :, z] > 0)
            for x, y in points[:: max(1, len(points) // 2500)]:
                draw.rectangle(
                    [int(x * sx), int(y * sy), int((x + 1) * sx) + 1, int((y + 1) * sy) + 1],
                    fill=(64, 220, 255, 110),
                )
        _draw_bbox_2d(draw, row, sx, sy, (64, 220, 255), width=2)

        if doctor_roi is not None and z < doctor_roi.shape[2]:
            doctor_points = np.argwhere(doctor_roi[:, :, z] > 0)
            for x, y in doctor_points[:: max(1, len(doctor_points) // 2500)]:
                draw.rectangle(
                    [int(x * sx), int(y * sy), int((x + 1) * sx) + 1, int((y + 1) * sy) + 1],
                    fill=(255, 64, 64, 95),
                )

        composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        text = (
            f"#{idx} {row.get('candidate_source')} z={z}\n"
            f"score={row.get('total_score')} bone={row.get('bone_overlap', row.get('pred_bone_overlap'))}\n"
            f"dist={row.get('bone_distance_mm')} cont={row.get('z_continuity_score', row.get('continuity_score'))}"
        )
        if "center_error_mm" in row:
            text += f"\nerr={row.get('center_error_mm')} cov={row.get('doctor_roi_coverage')}"
        ImageDraw.Draw(composed).multiline_text((8, 8), text, fill=(255, 255, 0), spacing=2)
        tiles.append(composed)

    columns = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (tile_size * columns, tile_size * rows), "black")
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_size
        y = (idx // columns) * tile_size
        sheet.paste(tile, (x, y))
    sheet.save(out_path)
