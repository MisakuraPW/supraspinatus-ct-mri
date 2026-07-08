from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti

from run_mri_viewer import MRSeries, choose_series, discover_mr_series, find_auto_mask, load_dicom_series


def normalize_mri_slice(slice_2d: np.ndarray) -> np.ndarray:
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    out = (slice_2d.astype(np.float32) - float(lo)) / max(float(hi - lo), 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def load_mask_for_series(series: MRSeries, image_shape: tuple[int, ...]) -> tuple[np.ndarray | None, Path | None, str]:
    mask_path = find_auto_mask(series.path)
    if mask_path is None:
        return None, None, "not_found"
    mask = load_nifti(mask_path).data
    if mask.shape == image_shape:
        return mask > 0, mask_path, "matched"
    if len(mask.shape) == 3 and mask.shape[0] == image_shape[1] and mask.shape[1] == image_shape[0] and mask.shape[2] == image_shape[2]:
        return np.transpose(mask, (1, 0, 2)) > 0, mask_path, "xy_transposed"
    return None, mask_path, f"shape_mismatch:{mask.shape}!={image_shape}"


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return None
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return int(mins[0]), int(mins[1]), int(mins[2]), int(maxs[0]), int(maxs[1]), int(maxs[2])


def compose_slice(
    image: np.ndarray,
    mask: np.ndarray | None,
    z: int,
    tile_size: int,
    title: str,
    draw_box: bool = True,
) -> Image.Image:
    gray = normalize_mri_slice(image[:, :, z])
    base = Image.fromarray(gray).convert("RGB").resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if mask is not None and z < mask.shape[2]:
        mask_slice = Image.fromarray((mask[:, :, z] > 0).astype(np.uint8) * 255).resize((tile_size, tile_size), Image.Resampling.NEAREST)
        color = Image.new("RGBA", base.size, (255, 48, 48, 0))
        alpha = mask_slice.point(lambda value: 110 if value > 0 else 0)
        color.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, color)
        draw = ImageDraw.Draw(overlay)

        if draw_box and np.any(mask[:, :, z] > 0):
            ys, xs = np.where(mask[:, :, z] > 0)
            sx = tile_size / image.shape[1]
            sy = tile_size / image.shape[0]
            box = [int(xs.min() * sx), int(ys.min() * sy), int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)]
            for offset in range(2):
                draw.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=(255, 230, 64, 255))

    composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    ImageDraw.Draw(composed).multiline_text((6, 6), title, fill=(255, 255, 0), spacing=2)
    return composed


def make_sheet(tiles: list[Image.Image], columns: int, fill: str = "black") -> Image.Image:
    if not tiles:
        raise ValueError("No tiles to render.")
    rows = int(np.ceil(len(tiles) / columns))
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * width, rows * height), fill)
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, columns)
        sheet.paste(tile, (col * width, row * height))
    return sheet


def crop_around_bbox(image: np.ndarray, mask: np.ndarray | None, bbox: tuple[int, int, int, int, int, int] | None, pad: int) -> tuple[np.ndarray, np.ndarray | None]:
    if bbox is None:
        return image, mask
    y1, x1, _z1, y2, x2, _z2 = bbox
    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(image.shape[0] - 1, y2 + pad)
    x2 = min(image.shape[1] - 1, x2 + pad)
    cropped_image = image[y1 : y2 + 1, x1 : x2 + 1, :]
    cropped_mask = None if mask is None else mask[y1 : y2 + 1, x1 : x2 + 1, :]
    return cropped_image, cropped_mask


def render_case(series: MRSeries, output_dir: Path, tile_size: int, focus_tile_size: int, focus_slices: int) -> dict[str, str]:
    image, _spacing, meta = load_dicom_series(series)
    mask, mask_path, mask_status = load_mask_for_series(series, image.shape)
    bbox = mask_bbox(mask) if mask is not None else None
    case_label = series.case
    desc = str(meta.get("series_description", ""))

    all_tiles = []
    for z in range(image.shape[2]):
        title = f"{case_label}\nz={z:02d} {desc}"
        all_tiles.append(compose_slice(image, mask, z, tile_size, title))
    all_path = output_dir / "cases" / f"{case_label}_mri_all_slices.png"
    all_path.parent.mkdir(parents=True, exist_ok=True)
    make_sheet(all_tiles, columns=6).save(all_path)

    if bbox is not None:
        center_z = int(round((bbox[2] + bbox[5]) / 2.0))
    else:
        center_z = image.shape[2] // 2
    half = max(1, focus_slices // 2)
    z_values = list(range(max(0, center_z - half), min(image.shape[2], center_z + half + 1)))
    cropped_image, cropped_mask = crop_around_bbox(image, mask, bbox, pad=42)
    focus_tiles = []
    for z in z_values:
        title = f"{case_label} focus\nz={z:02d} red=ROI"
        focus_tiles.append(compose_slice(cropped_image, cropped_mask, z, focus_tile_size, title))
    focus_path = output_dir / "cases" / f"{case_label}_mri_tendon_focus.png"
    make_sheet(focus_tiles, columns=len(focus_tiles)).save(focus_path)

    return {
        "case": case_label,
        "description": desc,
        "shape": f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}",
        "mask_status": mask_status,
        "mask_path": "" if mask_path is None else str(mask_path),
        "roi_voxels": "0" if mask is None else str(int(mask.sum())),
        "roi_bbox": "" if bbox is None else ",".join(str(v) for v in bbox),
        "all_slices_preview": str(all_path),
        "focus_preview": str(focus_path),
    }


def make_contact_sheet(paths: list[Path], output_path: Path, thumb_width: int, columns: int) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumbs.append(image.resize((thumb_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_sheet(thumbs, columns=columns).save(output_path)


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "description",
        "shape",
        "mask_status",
        "roi_voxels",
        "roi_bbox",
        "mask_path",
        "all_slices_preview",
        "focus_preview",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render static MRI previews with optional tendon ROI overlays.")
    parser.add_argument("--data-root", default="Data/label")
    parser.add_argument("--output-dir", default="outputs/2026-07-08_mri_tendon_previews")
    parser.add_argument("--case", action="append", help="Optional case filter. Can be passed multiple times.")
    parser.add_argument("--contains", default=None, help="Optional series description/protocol/path substring filter.")
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--focus-tile-size", type=int, default=320)
    parser.add_argument("--focus-slices", type=int, default=5)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    series_list = discover_mr_series(data_root)
    cases = sorted({item.case for item in series_list})
    if args.case:
        wanted = {case.lower() for case in args.case}
        cases = [case for case in cases if case.lower() in wanted]

    rows = []
    for case in cases:
        series = choose_series(series_list, case=case, series_index=None, contains=args.contains)
        row = render_case(series, output_dir, args.tile_size, args.focus_tile_size, args.focus_slices)
        rows.append(row)
        print(f"rendered {case}: {row['focus_preview']}")

    write_summary(rows, output_dir / "mri_tendon_preview_summary.csv")
    make_contact_sheet([Path(row["all_slices_preview"]) for row in rows], output_dir / "mri_all_cases_all_slices_contact_sheet.png", thumb_width=720, columns=2)
    make_contact_sheet([Path(row["focus_preview"]) for row in rows], output_dir / "mri_all_cases_tendon_focus_contact_sheet.png", thumb_width=720, columns=2)
    print(f"wrote {len(rows)} MRI cases to {output_dir}")


if __name__ == "__main__":
    main()
