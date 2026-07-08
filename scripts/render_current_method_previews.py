from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.localization.multi_bone_traditional import normalize_slice

from run_multibone_dicom_inference import choose_60kev_series, discover_case_dirs, load_dicom_series


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_first(paths, pred):
    for path in paths:
        if pred(path):
            return path
    raise FileNotFoundError("Required file was not found.")


def bbox_from_row(row: dict[str, str]) -> tuple[int, int, int, int, int, int]:
    return (
        int(float(row["pred_box_x1"])),
        int(float(row["pred_box_y1"])),
        int(float(row["pred_box_z1"])),
        int(float(row["pred_box_x2"])),
        int(float(row["pred_box_y2"])),
        int(float(row["pred_box_z2"])),
    )


def draw_bbox(draw: ImageDraw.ImageDraw, row: dict[str, str], scale_x: float, scale_y: float, color: tuple[int, int, int], width: int = 3) -> None:
    x1, y1, _z1, x2, y2, _z2 = bbox_from_row(row)
    box = [int(x1 * scale_x), int(y1 * scale_y), int((x2 + 1) * scale_x), int((y2 + 1) * scale_y)]
    for offset in range(width):
        draw.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=color)


def draw_mask_points(draw: ImageDraw.ImageDraw, mask: np.ndarray, z: int, scale_x: float, scale_y: float, color: tuple[int, int, int, int]) -> None:
    if z < 0 or z >= mask.shape[2]:
        return
    points = np.argwhere(mask[:, :, z] > 0)
    for x, y in points[:: max(1, len(points) // 3500)]:
        draw.rectangle(
            [int(x * scale_x), int(y * scale_y), int((x + 1) * scale_x) + 1, int((y + 1) * scale_y) + 1],
            fill=color,
        )


def render_case_preview(
    image: np.ndarray,
    row: dict[str, str],
    out_path: Path,
    doctor_roi: np.ndarray | None = None,
    title: str = "",
    tile_size: int = 512,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    center = int(round(float(row["pred_center_z"])))
    z_values = [max(0, center - 2), center, min(image.shape[2] - 1, center + 2)]
    tiles = []
    for z in z_values:
        base = Image.fromarray(normalize_slice(image[:, :, z].T)).convert("RGB").resize((tile_size, tile_size))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        sx = tile_size / image.shape[0]
        sy = tile_size / image.shape[1]
        draw_bbox(draw, row, sx, sy, (64, 220, 255), width=3)
        if doctor_roi is not None:
            draw_mask_points(draw, doctor_roi, z, sx, sy, (255, 64, 64, 95))
        composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        text_lines = [
            title,
            f"slice={z} source={row.get('candidate_source', row.get('selected_method', ''))}",
        ]
        if row.get("center_error_mm") not in ("", None):
            text_lines.append(f"err={row.get('center_error_mm')} cov={row.get('doctor_roi_coverage')} iou={row.get('bbox_iou', row.get('pred_box_doctor_bbox_iou', ''))}")
        if row.get("learning_score") not in ("", None):
            text_lines.append(f"score={row.get('learning_score')} bone={row.get('bone_overlap', row.get('pred_bone_overlap', ''))}")
        ImageDraw.Draw(composed).multiline_text((8, 8), "\n".join(text_lines), fill=(255, 255, 0), spacing=3)
        tiles.append(composed)

    sheet = Image.new("RGB", (tile_size * len(tiles), tile_size), "black")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, (idx * tile_size, 0))
    sheet.save(out_path)


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 640, columns: int = 2) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        image = image.resize((thumb_width, max(1, int(image.height * ratio))))
        thumbs.append(image)
    rows = int(np.ceil(len(thumbs) / columns))
    row_heights = []
    for row_idx in range(rows):
        row_tiles = thumbs[row_idx * columns : (row_idx + 1) * columns]
        row_heights.append(max(tile.height for tile in row_tiles))
    sheet = Image.new("RGB", (thumb_width * columns, sum(row_heights)), "black")
    y = 0
    for row_idx in range(rows):
        row_tiles = thumbs[row_idx * columns : (row_idx + 1) * columns]
        for col_idx, tile in enumerate(row_tiles):
            sheet.paste(tile, (col_idx * thumb_width, y))
        y += row_heights[row_idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def render_labeled(data_dir: Path, selections_csv: Path, output_dir: Path) -> list[Path]:
    out_paths = []
    rows = read_rows(selections_csv)
    for row in rows:
        case_dir = data_dir / row["case"] / "CT"
        ct_path = find_first(case_dir.iterdir(), lambda p: p.is_file() and "60" in p.name.lower() and ".nii" in p.name.lower())
        roi_path = find_first(case_dir.iterdir(), lambda p: p.is_file() and "roi" in p.name.lower() and ".nii" in p.name.lower())
        image = load_nifti(ct_path).data.astype(np.float32)
        doctor_roi = load_nifti(roi_path).data != 0
        out_path = output_dir / "labeled" / f"{row['case']}_current_method_preview.png"
        title = f"{row['case']} pred=cyan doctor=red"
        render_case_preview(image, row, out_path, doctor_roi=doctor_roi, title=title)
        out_paths.append(out_path)
    return out_paths


def render_unlabeled(data_dir: Path, selections_csv: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(selections_csv)
    case_dirs = {path.name: path for path in discover_case_dirs(data_dir)}
    out_paths = []
    for row in rows:
        case_dir = case_dirs[row["case"]]
        series_dir = choose_60kev_series(case_dir)
        image = load_dicom_series(series_dir).image.data.astype(np.float32)
        out_path = output_dir / "unlabeled" / f"{row['case']}_current_method_preview.png"
        title = f"{row['case']} pred=cyan"
        render_case_preview(image, row, out_path, doctor_roi=None, title=title)
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render current robust-ranker previews for labeled and unlabeled CT cases.")
    parser.add_argument("--labeled-data-dir", default="Data/label")
    parser.add_argument("--unlabeled-data-dir", default="Data/unlabel")
    parser.add_argument("--labeled-selections", default="outputs/2026-07_candidate_ranker_experiment/results/unified_learned_labeled_selections.csv")
    parser.add_argument("--unlabeled-selections", default="outputs/2026-07_candidate_ranker_experiment_with_visual_feedback/results/unlabeled_unified_selections.csv")
    parser.add_argument("--output-dir", default="outputs/2026-07-06_current_method_previews")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labeled_paths = render_labeled(Path(args.labeled_data_dir), Path(args.labeled_selections), output_dir)
    unlabeled_paths = render_unlabeled(Path(args.unlabeled_data_dir), Path(args.unlabeled_selections), output_dir)
    make_contact_sheet(labeled_paths, output_dir / "labeled_current_method_contact_sheet.png")
    make_contact_sheet(unlabeled_paths, output_dir / "unlabeled_current_method_contact_sheet.png")
    print(f"wrote {len(labeled_paths)} labeled previews and {len(unlabeled_paths)} unlabeled previews to {output_dir}")


if __name__ == "__main__":
    main()
