from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
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


def normalize_slice(slice_2d: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    arr = np.clip((slice_2d.astype(np.float32) - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask.astype(bool), iterations=int(iterations))
    except Exception:
        return mask.astype(bool)


def mask_outline(mask_2d: np.ndarray) -> np.ndarray:
    mask = mask_2d.astype(bool)
    if not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_erosion

        return mask & ~binary_erosion(mask)
    except Exception:
        return mask


def overlay_mask(base_gray: np.ndarray, mask_2d: np.ndarray, color: tuple[int, int, int], alpha: float = 0.32) -> Image.Image:
    rgb = np.repeat(base_gray[:, :, None], 3, axis=2).astype(np.float32)
    fill = mask_2d.astype(bool)
    if fill.any():
        rgb[fill] = (1.0 - alpha) * rgb[fill] + alpha * np.asarray(color, dtype=np.float32)
    outline = mask_outline(fill)
    rgb[outline] = np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def choose_slices(*masks: np.ndarray, max_slices: int = 6) -> list[int]:
    combined = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool)
    z_has = np.where(combined.reshape(-1, combined.shape[2]).any(axis=0))[0]
    if len(z_has) == 0:
        return list(np.linspace(0, combined.shape[2] - 1, max_slices, dtype=int))
    if len(z_has) <= max_slices:
        return [int(z) for z in z_has]
    return [int(z) for z in np.linspace(int(z_has[0]), int(z_has[-1]), max_slices, dtype=int)]


def make_case_preview(
    image: np.ndarray,
    morph_mask: np.ndarray,
    total_mask: np.ndarray,
    guided_mask: np.ndarray,
    out_path: Path,
    title: str,
    max_slices: int,
    window: tuple[float, float],
) -> None:
    z_values = choose_slices(morph_mask, total_mask, guided_mask, max_slices=max_slices)
    tile_w, tile_h = 360, 300
    labels = [
        ("CT", None, (255, 255, 255)),
        ("HU > threshold", morph_mask, (64, 220, 255)),
        ("TotalSeg shoulder bones", total_mask, (255, 215, 64)),
        ("TotalSeg-guided HU", guided_mask, (80, 235, 120)),
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
            tile.thumbnail((tile_w, tile_h - 28), Image.Resampling.BILINEAR)
            x = col * tile_w + (tile_w - tile.width) // 2
            y = row * tile_h + 24 + (tile_h - 28 - tile.height) // 2
            canvas.paste(tile, (x, y))
            draw.text((col * tile_w + 6, row * tile_h + 4), f"{title} | z={z} | {label}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview morphology vs TotalSegmentator shoulder-bone masks.")
    parser.add_argument("--data-dir", default="Data/label")
    parser.add_argument("--totalseg-mask-dir", default="outputs/2026-07_totalseg_shoulder_bones")
    parser.add_argument("--output-dir", default="outputs/2026-07_bone_segmentation_compare")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--bone-threshold", type=float, default=300.0)
    parser.add_argument("--guided-dilation-voxels", type=int, default=4)
    parser.add_argument("--max-slices", type=int, default=6)
    parser.add_argument("--window", nargs=2, type=float, default=(-100.0, 700.0))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    for case_dir in discover_cases(Path(args.data_dir), args.cases):
        ct_path = find_ct_60kev(case_dir)
        image = load_nifti(ct_path).data.astype(np.float32)
        morph = image > args.bone_threshold
        total_path = Path(args.totalseg_mask_dir) / case_dir.name / "shoulder_bones_combined.nii.gz"
        if total_path.exists():
            total = load_mask_compatible(total_path, image.shape)
        else:
            total = np.zeros_like(morph, dtype=bool)
        guided = morph & dilate_mask(total, args.guided_dilation_voxels)
        preview_path = output_dir / "previews" / f"{case_dir.name}_bone_compare.png"
        make_case_preview(
            image,
            morph,
            total,
            guided,
            preview_path,
            title=case_dir.name,
            max_slices=args.max_slices,
            window=tuple(args.window),
        )
        rows.append(
            {
                "case": case_dir.name,
                "ct_path": str(ct_path),
                "totalseg_mask_path": str(total_path),
                "morph_voxels": int(morph.sum()),
                "totalseg_voxels": int(total.sum()),
                "guided_voxels": int(guided.sum()),
                "guided_to_morph_fraction": round(float(guided.sum() / max(1, morph.sum())), 5),
                "guided_to_totalseg_fraction": round(float(guided.sum() / max(1, total.sum())), 5),
                "preview_path": str(preview_path),
            }
        )
        print(f"{case_dir.name}: morph={int(morph.sum())} total={int(total.sum())} guided={int(guided.sum())}")

    summary_path = output_dir / "bone_segmentation_compare_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
