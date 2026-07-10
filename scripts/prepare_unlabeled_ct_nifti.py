from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.dicom_io import load_dicom_volume, scan_dicom_series
from supraspinatus_locator.data.nifti_io import save_nifti_like


def discover_cases(data_dir: Path, case_names: list[str] | None) -> list[Path]:
    cases = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if case_names:
        wanted = set(case_names)
        cases = [path for path in cases if path.name in wanted]
    return cases


def discover_dicom_series(case_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in case_dir.rglob("*") if p.is_dir()):
        files = [p for p in path.iterdir() if p.is_file()]
        if not files:
            continue
        try:
            series = scan_dicom_series(path)
        except Exception:
            continue
        meta = series.metadata
        modality = str(meta.get("modality", "")).upper()
        count = int(meta.get("count", 0) or 0)
        rows.append(
            {
                "path": path,
                "modality": modality,
                "count": count,
                "rows": int(meta.get("rows", 0) or 0),
                "columns": int(meta.get("columns", 0) or 0),
                "series_description": str(meta.get("series_description", "")),
                "protocol_name": str(meta.get("protocol_name", "")),
                "slice_thickness": float(meta.get("slice_thickness", 0.0) or 0.0),
            }
        )
    return rows


def score_series(row: dict[str, object]) -> tuple[int, int, int, float]:
    modality_bonus = 1000000 if row.get("modality") == "CT" else 0
    count = int(row.get("count", 0) or 0)
    area = int(row.get("rows", 0) or 0) * int(row.get("columns", 0) or 0)
    thickness = float(row.get("slice_thickness", 99.0) or 99.0)
    return (modality_bonus, count, area, -thickness)


def normalize_slice(slice_2d: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    arr = np.clip((slice_2d.astype(np.float32) - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def make_preview(volume: np.ndarray, out_path: Path, title: str, max_slices: int, window: tuple[float, float]) -> None:
    z_values = [int(z) for z in np.linspace(0, volume.shape[2] - 1, min(max_slices, volume.shape[2]), dtype=int)]
    tile_w, tile_h = 320, 280
    canvas = Image.new("RGB", (tile_w * len(z_values), tile_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for col, z in enumerate(z_values):
        base = normalize_slice(volume[:, :, z], window)
        tile = Image.fromarray(np.repeat(base[:, :, None], 3, axis=2))
        tile.thumbnail((tile_w, tile_h - 28), Image.Resampling.BILINEAR)
        x = col * tile_w + (tile_w - tile.width) // 2
        y = 24 + (tile_h - 28 - tile.height) // 2
        canvas.paste(tile, (x, y))
        draw.text((col * tile_w + 6, 4), f"{title} | z={z}", fill=(255, 255, 96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


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


def process_case(args: argparse.Namespace, case_dir: Path, output_dir: Path) -> dict[str, object]:
    series_rows = discover_dicom_series(case_dir)
    ct_rows = [row for row in series_rows if row.get("modality") == "CT" and int(row.get("count", 0) or 0) >= args.min_slices]
    candidates = ct_rows or [row for row in series_rows if int(row.get("count", 0) or 0) >= args.min_slices]
    if not candidates:
        return {
            "case": case_dir.name,
            "status": "failed_no_dicom_ct_series",
            "series_count": len(series_rows),
            "selected_series": "",
        }
    selected = max(candidates, key=score_series)
    volume, spacing, meta = load_dicom_volume(Path(selected["path"]))
    raw_spacing = tuple(float(v) for v in spacing)
    selected_thickness = float(selected.get("slice_thickness", 0.0) or 0.0)
    z_spacing_source = "dicom_position"
    if spacing[2] <= 0 or spacing[2] > args.max_z_spacing_mm or (selected_thickness > 0 and spacing[2] > selected_thickness * 5.0):
        fallback_z = selected_thickness if selected_thickness > 0 else 1.0
        spacing = (float(spacing[0]), float(spacing[1]), float(fallback_z))
        z_spacing_source = "slice_thickness_fallback"
    case_out = output_dir / case_dir.name / "CT"
    nii_path = case_out / "60kev.nii.gz"
    save_nifti_like(nii_path, volume.astype(np.float32), reference=None, spacing=spacing)
    preview_path = output_dir / "_previews" / f"{case_dir.name}_ct_nifti_preview.png"
    make_preview(volume, preview_path, case_dir.name, int(args.max_slices), tuple(args.window))
    return {
        "case": case_dir.name,
        "status": "ok",
        "series_count": len(series_rows),
        "selected_series": str(selected["path"]),
        "selected_modality": selected.get("modality", ""),
        "selected_count": selected.get("count", ""),
        "selected_rows": selected.get("rows", ""),
        "selected_columns": selected.get("columns", ""),
        "selected_slice_thickness": selected.get("slice_thickness", ""),
        "selected_series_description": selected.get("series_description", ""),
        "selected_protocol_name": selected.get("protocol_name", ""),
        "shape": "x".join(str(v) for v in volume.shape),
        "raw_spacing": "x".join(f"{float(v):.4f}" for v in raw_spacing),
        "spacing": "x".join(f"{float(v):.4f}" for v in spacing),
        "z_spacing_source": z_spacing_source,
        "min_hu": round(float(np.min(volume)), 4),
        "max_hu": round(float(np.max(volume)), 4),
        "mean_hu": round(float(np.mean(volume)), 4),
        "output_nifti": str(nii_path),
        "preview_path": str(preview_path),
        "metadata_modality": meta.get("modality", ""),
        "metadata_series_description": meta.get("series_description", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert unlabeled DICOM CT series into project-compatible NIfTI case folders.")
    parser.add_argument("--data-dir", default="Data/unlabel")
    parser.add_argument("--output-dir", default="outputs/2026-07_unlabel_ct_nifti")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--min-slices", type=int, default=20)
    parser.add_argument("--max-z-spacing-mm", type=float, default=8.0)
    parser.add_argument("--max-slices", type=int, default=5)
    parser.add_argument("--window", nargs=2, type=float, default=(-100.0, 700.0))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    output_dir = Path(args.output_dir)
    for case_dir in discover_cases(Path(args.data_dir), args.cases):
        row = process_case(args, case_dir, output_dir)
        rows.append(row)
        print(f"{row['case']}: {row['status']} selected={row.get('selected_series', '')}")
    write_csv(output_dir / "unlabeled_ct_nifti_summary.csv", rows)
    print(f"wrote prepared NIfTI cases to {output_dir}")


if __name__ == "__main__":
    main()
