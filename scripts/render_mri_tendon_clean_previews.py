from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti

from run_mri_viewer import MRSeries, choose_series, discover_mr_series, find_auto_mask, load_dicom_series


def normalize_mri(slice_2d: np.ndarray) -> np.ndarray:
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    out = (slice_2d.astype(np.float32) - float(lo)) / max(float(hi - lo), 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def load_roi(series: MRSeries, image_shape: tuple[int, ...]) -> tuple[np.ndarray | None, Path | None, str]:
    mask_path = find_auto_mask(series.path)
    if mask_path is None:
        return None, None, "not_found"
    mask = load_nifti(mask_path).data > 0
    if mask.shape == image_shape:
        return mask, mask_path, "matched"
    if len(mask.shape) == 3 and mask.shape[:2] == (image_shape[1], image_shape[0]) and mask.shape[2] == image_shape[2]:
        return np.transpose(mask, (1, 0, 2)), mask_path, "xy_transposed"
    return None, mask_path, f"shape_mismatch:{mask.shape}!={image_shape}"


def roi_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int, int, int] | None:
    if mask is None or not np.any(mask):
        return None
    pts = np.argwhere(mask)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return int(mn[0]), int(mn[1]), int(mn[2]), int(mx[0]), int(mx[1]), int(mx[2])


def crop_to_bbox(image: np.ndarray, mask: np.ndarray | None, bbox: tuple[int, int, int, int, int, int] | None, pad: int) -> tuple[np.ndarray, np.ndarray | None]:
    if bbox is None:
        return image, mask
    row1, col1, _z1, row2, col2, _z2 = bbox
    row1 = max(0, row1 - pad)
    col1 = max(0, col1 - pad)
    row2 = min(image.shape[0] - 1, row2 + pad)
    col2 = min(image.shape[1] - 1, col2 + pad)
    cropped_image = image[row1 : row2 + 1, col1 : col2 + 1, :]
    cropped_mask = None if mask is None else mask[row1 : row2 + 1, col1 : col2 + 1, :]
    return cropped_image, cropped_mask


def mask_boundary(mask_2d: np.ndarray) -> np.ndarray:
    m = mask_2d.astype(bool)
    if not np.any(m):
        return m
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(m, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            eroded &= padded[1 + dr : 1 + dr + m.shape[0], 1 + dc : 1 + dc + m.shape[1]]
    boundary = m & ~eroded
    thick = boundary.copy()
    padded_b = np.pad(boundary, 1, mode="constant", constant_values=False)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            thick |= padded_b[1 + dr : 1 + dr + m.shape[0], 1 + dc : 1 + dc + m.shape[1]]
    return thick


def make_base(slice_2d: np.ndarray, size: int, enhanced: bool) -> Image.Image:
    gray = normalize_mri(slice_2d)
    image = Image.fromarray(gray).convert("L")
    if enhanced:
        image = ImageOps.autocontrast(image, cutoff=1).filter(ImageFilter.SHARPEN)
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def render_panel(
    image: np.ndarray,
    mask: np.ndarray | None,
    z: int,
    size: int,
    title: str,
    mode: str,
) -> Image.Image:
    base = make_base(image[:, :, z], size, enhanced=(mode == "enhanced"))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if mode == "outline" and mask is not None and z < mask.shape[2]:
        boundary = mask_boundary(mask[:, :, z])
        if np.any(boundary):
            boundary_img = Image.fromarray(boundary.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
            color = Image.new("RGBA", base.size, (80, 255, 120, 0))
            color.putalpha(boundary_img.point(lambda value: 230 if value > 0 else 0))
            overlay = Image.alpha_composite(overlay, color)
            ys, xs = np.where(mask[:, :, z] > 0)
            sx = size / image.shape[1]
            sy = size / image.shape[0]
            box = [int(xs.min() * sx), int(ys.min() * sy), int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)]
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(box, outline=(255, 230, 64, 255), width=1)
    composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    ImageDraw.Draw(composed).multiline_text((6, 6), title, fill=(255, 255, 0), spacing=2)
    return composed


def render_case(series: MRSeries, output_dir: Path, size: int, pad: int, slices: int) -> dict[str, str]:
    image, _spacing, meta = load_dicom_series(series)
    mask, mask_path, mask_status = load_roi(series, image.shape)
    bbox = roi_bbox(mask)
    crop_image, crop_mask = crop_to_bbox(image, mask, bbox, pad=pad)
    center_z = image.shape[2] // 2 if bbox is None else int(round((bbox[2] + bbox[5]) / 2.0))
    half = max(1, slices // 2)
    z_values = list(range(max(0, center_z - half), min(image.shape[2], center_z + half + 1)))

    rows = []
    for z in z_values:
        raw = render_panel(crop_image, crop_mask, z, size, f"{series.case} raw\nz={z:02d}", "raw")
        enhanced = render_panel(crop_image, crop_mask, z, size, f"{series.case} enhanced\nz={z:02d}", "enhanced")
        outline = render_panel(crop_image, crop_mask, z, size, f"{series.case} ROI outline\nz={z:02d}", "outline")
        row = Image.new("RGB", (size * 3, size), "black")
        row.paste(raw, (0, 0))
        row.paste(enhanced, (size, 0))
        row.paste(outline, (size * 2, 0))
        rows.append(row)

    sheet = Image.new("RGB", (size * 3, size * len(rows)), "black")
    for idx, row in enumerate(rows):
        sheet.paste(row, (0, idx * size))
    out_path = output_dir / "cases" / f"{series.case}_mri_clean_tendon_guide.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)

    return {
        "case": series.case,
        "description": str(meta.get("series_description", "")),
        "shape": f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}",
        "mask_status": mask_status,
        "mask_path": "" if mask_path is None else str(mask_path),
        "roi_voxels": "0" if mask is None else str(int(mask.sum())),
        "preview": str(out_path),
    }


def make_contact_sheet(paths: list[Path], output_path: Path, thumb_width: int, columns: int) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumbs.append(image.resize((thumb_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS))
    if not thumbs:
        return
    tile_w = max(tile.width for tile in thumbs)
    tile_h = max(tile.height for tile in thumbs)
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), "black")
    for idx, tile in enumerate(thumbs):
        row, col = divmod(idx, columns)
        sheet.paste(tile, (col * tile_w, row * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = ["case", "description", "shape", "mask_status", "roi_voxels", "mask_path", "preview"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clean MRI tendon guide previews without filled ROI overlays.")
    parser.add_argument("--data-root", default="Data/label")
    parser.add_argument("--output-dir", default="outputs/2026-07-08_mri_tendon_clean_previews")
    parser.add_argument("--case", action="append")
    parser.add_argument("--contains")
    parser.add_argument("--size", type=int, default=260)
    parser.add_argument("--pad", type=int, default=52)
    parser.add_argument("--slices", type=int, default=5)
    args = parser.parse_args()

    series_list = discover_mr_series(args.data_root)
    cases = sorted({item.case for item in series_list})
    if args.case:
        wanted = {case.lower() for case in args.case}
        cases = [case for case in cases if case.lower() in wanted]

    output_dir = Path(args.output_dir)
    rows = []
    for case in cases:
        series = choose_series(series_list, case=case, series_index=None, contains=args.contains)
        row = render_case(series, output_dir, args.size, args.pad, args.slices)
        rows.append(row)
        print(f"rendered {case}: {row['preview']}")

    write_summary(rows, output_dir / "mri_clean_tendon_preview_summary.csv")
    make_contact_sheet([Path(row["preview"]) for row in rows], output_dir / "mri_all_cases_clean_tendon_contact_sheet.png", thumb_width=780, columns=2)
    print(f"wrote {len(rows)} cases to {output_dir}")


if __name__ == "__main__":
    main()
