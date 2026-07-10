from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import NiftiImage, load_nifti, save_nifti_like
from supraspinatus_locator.preprocessing.totalseg_bones import load_mask_compatible


def find_ct_60kev(case_dir: Path) -> Path:
    ct_dir = case_dir / "CT"
    matches = sorted(path for path in ct_dir.iterdir() if path.is_file() and "60" in path.name.lower() and ".nii" in path.name.lower())
    if not matches:
        raise FileNotFoundError(f"No 60keV NIfTI found under {ct_dir}")
    return matches[0]


def discover_cases(data_dir: Path, case_names: list[str] | None) -> list[Path]:
    cases = []
    for path in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not (path / "CT").is_dir():
            continue
        try:
            find_ct_60kev(path)
        except FileNotFoundError:
            continue
        cases.append(path)
    if case_names:
        wanted = set(case_names)
        cases = [path for path in cases if path.name in wanted]
    return cases


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    mask = mask.astype(bool)
    if iterations <= 0 or not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, iterations=int(iterations))
    except Exception:
        return mask


def mask_outline(mask_2d: np.ndarray) -> np.ndarray:
    mask = mask_2d.astype(bool)
    if not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_erosion

        return mask & ~binary_erosion(mask)
    except Exception:
        return mask


def clean_components(mask: np.ndarray, min_component_voxels: int, max_components: int) -> tuple[np.ndarray, int, int, float]:
    mask = mask.astype(bool)
    if not mask.any():
        return mask, 0, 0, 0.0
    try:
        from scipy.ndimage import label

        labels, count = label(mask)
        if count == 0:
            return mask, 0, 0, 0.0
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        component_ids = [int(idx) for idx in np.argsort(sizes)[::-1] if sizes[idx] >= min_component_voxels]
        kept_ids = component_ids[:max_components]
        cleaned = np.isin(labels, kept_ids)
        largest = int(sizes[kept_ids[0]]) if kept_ids else 0
        largest_fraction = float(largest / max(1, int(mask.sum())))
        return cleaned, int(count), len(kept_ids), largest_fraction
    except Exception:
        return mask, 1, 1, 1.0


def bbox_stats(mask: np.ndarray) -> dict[str, object]:
    pts = np.argwhere(mask.astype(bool))
    if len(pts) == 0:
        return {
            "bbox_min_x": "",
            "bbox_min_y": "",
            "bbox_min_z": "",
            "bbox_max_x": "",
            "bbox_max_y": "",
            "bbox_max_z": "",
            "z_span": 0,
        }
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return {
        "bbox_min_x": int(mins[0]),
        "bbox_min_y": int(mins[1]),
        "bbox_min_z": int(mins[2]),
        "bbox_max_x": int(maxs[0]),
        "bbox_max_y": int(maxs[1]),
        "bbox_max_z": int(maxs[2]),
        "z_span": int(maxs[2] - mins[2] + 1),
    }


def normalize_slice(slice_2d: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    arr = np.clip((slice_2d.astype(np.float32) - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def overlay_mask(base_gray: np.ndarray, mask_2d: np.ndarray, color: tuple[int, int, int], alpha: float = 0.30) -> Image.Image:
    rgb = np.repeat(base_gray[:, :, None], 3, axis=2).astype(np.float32)
    fill = mask_2d.astype(bool)
    if fill.any():
        rgb[fill] = (1.0 - alpha) * rgb[fill] + alpha * np.asarray(color, dtype=np.float32)
    outline = mask_outline(fill)
    rgb[outline] = np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def choose_slices(image: np.ndarray, *masks: np.ndarray, max_slices: int) -> list[int]:
    combined = np.zeros(image.shape, dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool)
    z_has = np.where(combined.reshape(-1, combined.shape[2]).any(axis=0))[0]
    if len(z_has) == 0:
        return list(np.linspace(0, image.shape[2] - 1, max_slices, dtype=int))
    if len(z_has) <= max_slices:
        return [int(z) for z in z_has]
    return [int(z) for z in np.linspace(int(z_has[0]), int(z_has[-1]), max_slices, dtype=int)]


def make_preview(
    image: np.ndarray,
    hu_mask: np.ndarray,
    total_union: np.ndarray,
    guided_hu: np.ndarray,
    fused: np.ndarray,
    out_path: Path,
    case_name: str,
    status: str,
    max_slices: int,
    window: tuple[float, float],
) -> None:
    z_values = choose_slices(image, hu_mask, total_union, guided_hu, fused, max_slices=max_slices)
    tile_w, tile_h = 340, 280
    labels = [
        ("CT", None, (255, 255, 255)),
        ("HU threshold", hu_mask, (64, 220, 255)),
        ("TotalSeg prior", total_union, (255, 215, 64)),
        ("TotalSeg-guided HU", guided_hu, (80, 235, 120)),
        ("final fused mask", fused, (255, 96, 96)),
    ]
    canvas = Image.new("RGB", (tile_w * len(labels), tile_h * len(z_values)), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for row, z in enumerate(z_values):
        base = normalize_slice(image[:, :, z].T, window)
        for col, (label, mask, color) in enumerate(labels):
            if mask is None:
                tile = Image.fromarray(np.repeat(base[:, :, None], 3, axis=2))
            else:
                tile = overlay_mask(base, mask[:, :, z].T, color)
            tile.thumbnail((tile_w, tile_h - 30), Image.Resampling.BILINEAR)
            x = col * tile_w + (tile_w - tile.width) // 2
            y = row * tile_h + 26 + (tile_h - 30 - tile.height) // 2
            canvas.paste(tile, (x, y))
            draw.text((col * tile_w + 6, row * tile_h + 4), f"{case_name} | {status} | z={z} | {label}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_totalseg_union(mask_dirs: list[Path], case_name: str, image_shape: tuple[int, ...], filename: str) -> tuple[np.ndarray, list[str], list[int]]:
    union = np.zeros(image_shape, dtype=bool)
    loaded: list[str] = []
    voxel_counts: list[int] = []
    for mask_dir in mask_dirs:
        mask_path = mask_dir / case_name / filename
        if not mask_path.exists():
            voxel_counts.append(0)
            continue
        mask = load_mask_compatible(mask_path, image_shape)
        union |= mask
        loaded.append(str(mask_path))
        voxel_counts.append(int(mask.sum()))
    return union, loaded, voxel_counts


def classify_quality(
    union_voxels: int,
    guided_voxels: int,
    morph_voxels: int,
    kept_components: int,
    z_span: int,
    min_totalseg_voxels: int,
    min_fused_voxels: int,
) -> tuple[str, float, str]:
    guided_fraction = float(guided_voxels / max(1, morph_voxels))
    score = 0.0
    reasons: list[str] = []
    if union_voxels >= min_totalseg_voxels:
        score += 1.0
    else:
        reasons.append("totalseg_too_small")
    if guided_voxels >= min_fused_voxels:
        score += 1.0
    else:
        reasons.append("guided_hu_too_small")
    if 0.02 <= guided_fraction <= 0.70:
        score += 1.0
    else:
        reasons.append("guided_fraction_out_of_range")
    if 1 <= kept_components <= 12:
        score += 0.7
    else:
        reasons.append("component_count_unstable")
    if z_span >= 8:
        score += 0.5
    else:
        reasons.append("z_span_too_short")

    if union_voxels < min_totalseg_voxels or guided_voxels < min_fused_voxels:
        return "invalid_fallback_to_hu", round(score, 4), ";".join(reasons)
    if score < 3.0:
        return "weak_use_with_review", round(score, 4), ";".join(reasons)
    return "usable", round(score, 4), ";".join(reasons) or "ok"


def process_case(args: argparse.Namespace, case_dir: Path, output_dir: Path) -> dict[str, object]:
    ct_path = find_ct_60kev(case_dir)
    image_obj: NiftiImage = load_nifti(ct_path)
    image = image_obj.data.astype(np.float32)
    hu_mask = image > float(args.bone_threshold)
    total_union, loaded_paths, voxel_counts = load_totalseg_union(
        [Path(p) for p in args.totalseg_mask_dirs],
        case_dir.name,
        image.shape,
        args.totalseg_mask_filename,
    )
    total_prior = dilate_mask(total_union, int(args.prior_dilation_voxels))
    guided_hu = hu_mask & dilate_mask(total_union, int(args.guided_dilation_voxels))
    cleaned_guided, raw_components, kept_components, largest_fraction = clean_components(
        guided_hu,
        int(args.min_component_voxels),
        int(args.max_components),
    )
    bbox = bbox_stats(cleaned_guided)
    status, quality_score, quality_reason = classify_quality(
        int(total_union.sum()),
        int(cleaned_guided.sum()),
        int(hu_mask.sum()),
        kept_components,
        int(bbox["z_span"]),
        int(args.min_totalseg_voxels),
        int(args.min_fused_voxels),
    )
    fused = cleaned_guided if status != "invalid_fallback_to_hu" else np.zeros_like(cleaned_guided, dtype=bool)
    case_out = output_dir / case_dir.name
    case_out.mkdir(parents=True, exist_ok=True)
    save_nifti_like(case_out / "shoulder_bones_totalseg_prior_union.nii.gz", total_union.astype(np.uint8), reference=image_obj)
    save_nifti_like(case_out / "shoulder_bones_totalseg_prior_dilated.nii.gz", total_prior.astype(np.uint8), reference=image_obj)
    save_nifti_like(case_out / "shoulder_bones_hu_threshold.nii.gz", hu_mask.astype(np.uint8), reference=image_obj)
    save_nifti_like(case_out / "shoulder_bones_totalseg_guided_hu_raw.nii.gz", guided_hu.astype(np.uint8), reference=image_obj)
    save_nifti_like(case_out / "shoulder_bones_fused_hu.nii.gz", fused.astype(np.uint8), reference=image_obj)
    preview_path = output_dir / "previews" / f"{case_dir.name}_fusion_preview.png"
    make_preview(
        image,
        hu_mask,
        total_union,
        guided_hu,
        fused,
        preview_path,
        case_name=case_dir.name,
        status=status,
        max_slices=int(args.max_slices),
        window=tuple(args.window),
    )
    return {
        "case": case_dir.name,
        "status": status,
        "quality_score": quality_score,
        "quality_reason": quality_reason,
        "ct_path": str(ct_path),
        "loaded_totalseg_masks": " | ".join(loaded_paths),
        "totalseg_source_voxels": " ".join(str(v) for v in voxel_counts),
        "hu_threshold_voxels": int(hu_mask.sum()),
        "totalseg_union_voxels": int(total_union.sum()),
        "totalseg_prior_dilated_voxels": int(total_prior.sum()),
        "guided_hu_raw_voxels": int(guided_hu.sum()),
        "fused_hu_voxels": int(fused.sum()),
        "guided_to_hu_fraction": round(float(guided_hu.sum() / max(1, hu_mask.sum())), 6),
        "fused_to_hu_fraction": round(float(fused.sum() / max(1, hu_mask.sum())), 6),
        "raw_component_count": raw_components,
        "kept_component_count": kept_components,
        "largest_component_fraction": round(largest_fraction, 6),
        "preview_path": str(preview_path),
        **bbox,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    total = len(rows)
    usable = sum(1 for row in rows if row.get("status") == "usable")
    weak = sum(1 for row in rows if row.get("status") == "weak_use_with_review")
    invalid = sum(1 for row in rows if row.get("status") == "invalid_fallback_to_hu")
    lines = [
        "# TotalSeg + HU 骨分割融合报告",
        "",
        "本脚本不训练模型，只做 TotalSegmentator 推理结果的质量评分与后处理融合。",
        "",
        "## 方法",
        "",
        "- `TotalSeg prior`：把一个或多个 TotalSeg 运行结果合并为肩部骨骼空间先验。",
        "- `HU threshold`：使用 CT HU 阈值提取皮质骨候选，默认阈值为 300 HU。",
        "- `TotalSeg-guided HU`：在 TotalSeg 先验邻域内保留 HU 骨体素，用深度学习结果限制空间范围，用 HU 保留骨皮质边缘。",
        "- `final fused mask`：对 guided HU 做连通域清理；若 TotalSeg 先验过小或融合体素过少，则输出空 mask，让后续定位脚本自动回退到纯 HU。",
        "",
        "## 参数",
        "",
        f"- TotalSeg mask dirs: `{', '.join(args.totalseg_mask_dirs)}`",
        f"- bone threshold HU: `{args.bone_threshold}`",
        f"- guided dilation voxels: `{args.guided_dilation_voxels}`",
        f"- min TotalSeg voxels: `{args.min_totalseg_voxels}`",
        f"- min fused voxels: `{args.min_fused_voxels}`",
        "",
        "## 质量概览",
        "",
        f"- cases: `{total}`",
        f"- usable: `{usable}`",
        f"- weak_use_with_review: `{weak}`",
        f"- invalid_fallback_to_hu: `{invalid}`",
        "",
        "## 后续定位推荐命令",
        "",
        "```bash",
        "python scripts/run_totalseg_locator_strategy_sweep.py \\",
        "  --data-dir Data/label \\",
        f"  --bone-mask-dir {args.output_dir} \\",
        "  --bone-mask-filename shoulder_bones_fused_hu.nii.gz \\",
        "  --output-root outputs/2026-07_totalseg_fused_locator_sweep \\",
        "  --quick",
        "```",
        "",
        "如果某例状态为 `invalid_fallback_to_hu`，`shoulder_bones_fused_hu.nii.gz` 会是空 mask，定位脚本配合 `--allow-threshold-bone-fallback` 会自动回到纯 HU。",
        "",
        "## 逐例结果",
        "",
        "| case | status | score | TotalSeg voxels | guided HU | fused HU | reason | preview |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['status']} | {row['quality_score']} | "
            f"{row['totalseg_union_voxels']} | {row['guided_hu_raw_voxels']} | {row['fused_hu_voxels']} | "
            f"{row['quality_reason']} | {row['preview_path']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse TotalSegmentator shoulder-bone priors with HU cortical bone masks.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--totalseg-mask-dirs", nargs="+", required=True, help="One or more TotalSeg output roots.")
    parser.add_argument("--output-dir", default="outputs/2026-07_totalseg_hu_fused_bones")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--totalseg-mask-filename", default="shoulder_bones_combined.nii.gz")
    parser.add_argument("--bone-threshold", type=float, default=300.0)
    parser.add_argument("--guided-dilation-voxels", type=int, default=4)
    parser.add_argument("--prior-dilation-voxels", type=int, default=1)
    parser.add_argument("--min-totalseg-voxels", type=int, default=10000)
    parser.add_argument("--min-fused-voxels", type=int, default=10000)
    parser.add_argument("--min-component-voxels", type=int, default=250)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--max-slices", type=int, default=6)
    parser.add_argument("--window", nargs=2, type=float, default=(-100.0, 700.0))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    for case_dir in discover_cases(Path(args.data_dir), args.cases):
        row = process_case(args, case_dir, output_dir)
        rows.append(row)
        print(
            f"{row['case']}: {row['status']} score={row['quality_score']} "
            f"total={row['totalseg_union_voxels']} fused={row['fused_hu_voxels']}"
        )
    write_csv(output_dir / "totalseg_hu_fusion_summary.csv", rows)
    write_report(output_dir / "reports" / "totalseg_hu_fusion_report.md", rows, args)
    print(f"wrote fusion outputs to {output_dir}")


if __name__ == "__main__":
    main()
