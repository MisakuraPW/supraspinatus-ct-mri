from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.localization.multi_bone_traditional import erode_binary_2d, normalize_slice

from run_multibone_dicom_inference import choose_60kev_series, discover_case_dirs, load_dicom_series


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_bone_edge_rows(rows: list[dict[str, str]], topk: int) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("candidate_source") == "bone_edge_tendon":
            groups[row["case"]].append(row)
    out = {}
    for case, case_rows in groups.items():
        case_rows.sort(key=lambda row: float(row.get("total_score") or 0.0), reverse=True)
        out[case] = case_rows[:topk]
    return out


def find_first(paths, pred):
    for path in paths:
        if pred(path):
            return path
    raise FileNotFoundError("Required file was not found.")


def draw_bbox(draw: ImageDraw.ImageDraw, row: dict[str, str], sx: float, sy: float, color: tuple[int, int, int], width: int = 3) -> None:
    x1 = int(float(row["pred_box_x1"]) * sx)
    y1 = int(float(row["pred_box_y1"]) * sy)
    x2 = int((float(row["pred_box_x2"]) + 1.0) * sx)
    y2 = int((float(row["pred_box_y2"]) + 1.0) * sy)
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def draw_mask_points(draw: ImageDraw.ImageDraw, mask: np.ndarray, z: int, sx: float, sy: float, color: tuple[int, int, int, int]) -> None:
    if z < 0 or z >= mask.shape[2]:
        return
    points = np.argwhere(mask[:, :, z] > 0)
    for x, y in points[:: max(1, len(points) // 3500)]:
        draw.rectangle(
            [int(x * sx), int(y * sy), int((x + 1) * sx) + 1, int((y + 1) * sy) + 1],
            fill=color,
        )


def render_case(
    image: np.ndarray,
    rows: list[dict[str, str]],
    out_path: Path,
    doctor_roi: np.ndarray | None = None,
    tile_size: int = 384,
) -> None:
    if not rows:
        return
    tiles = []
    for idx, row in enumerate(rows, start=1):
        z = int(round(float(row["pred_center_z"])))
        z = max(0, min(image.shape[2] - 1, z))
        base = Image.fromarray(normalize_slice(image[:, :, z].T)).convert("RGB").resize((tile_size, tile_size))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        sx = tile_size / image.shape[0]
        sy = tile_size / image.shape[1]

        bone_slice = image[:, :, z].T > 300.0
        edge_slice = bone_slice & ~erode_binary_2d(bone_slice)
        edge_y, edge_x = np.nonzero(edge_slice)
        for x, y in zip(edge_x[:: max(1, len(edge_x) // 4500)], edge_y[:: max(1, len(edge_y) // 4500)]):
            draw.point((int(x * sx), int(y * sy)), fill=(120, 220, 255, 150))

        if row.get("edge_point_x") not in ("", None) and row.get("edge_point_y") not in ("", None):
            ex = int(float(row["edge_point_x"]) * sx)
            ey = int(float(row["edge_point_y"]) * sy)
            draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(0, 255, 80, 230))
            if row.get("surface_normal_x") not in ("", None):
                nx = float(row["surface_normal_x"])
                ny = float(row["surface_normal_y"])
                draw.line([ex, ey, int(ex + nx * 34), int(ey + ny * 34)], fill=(0, 255, 80, 230), width=3)

        draw_bbox(draw, row, sx, sy, (255, 230, 64), width=3)
        if doctor_roi is not None:
            draw_mask_points(draw, doctor_roi, z, sx, sy, (255, 64, 64, 90))

        composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        text = (
            f"#{idx} {row['case']} bone_edge z={z}\n"
            f"score={row.get('total_score')} bone={row.get('bone_overlap')}\n"
            f"angle={row.get('edge_angle_deg')} dist={row.get('surface_distance_mm')}\n"
        )
        if row.get("center_error_mm") not in ("", None):
            text += f"err={row.get('center_error_mm')} cov={row.get('doctor_roi_coverage')}"
        ImageDraw.Draw(composed).multiline_text((8, 8), text, fill=(255, 255, 0), spacing=2)
        tiles.append(composed)

    cols = min(4, len(tiles))
    rows_count = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (tile_size * cols, tile_size * rows_count), "black")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * tile_size, (idx // cols) * tile_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_width: int = 620, cols: int = 2) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        ratio = thumb_width / im.width
        thumbs.append(im.resize((thumb_width, max(1, int(im.height * ratio)))))
    row_heights = []
    for idx in range(0, len(thumbs), cols):
        row_heights.append(max(im.height for im in thumbs[idx : idx + cols]))
    sheet = Image.new("RGB", (thumb_width * cols, sum(row_heights)), "black")
    y = 0
    for idx in range(0, len(thumbs), cols):
        row = thumbs[idx : idx + cols]
        for col, im in enumerate(row):
            sheet.paste(im, (col * thumb_width, y))
        y += row_heights[idx // cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def render_labeled(data_dir: Path, candidates_csv: Path, out_dir: Path, topk: int) -> list[Path]:
    grouped = group_bone_edge_rows(read_rows(candidates_csv), topk)
    out_paths = []
    for case, rows in sorted(grouped.items()):
        ct_dir = data_dir / case / "CT"
        ct_path = find_first(ct_dir.iterdir(), lambda p: p.is_file() and "60" in p.name.lower() and ".nii" in p.name.lower())
        roi_path = find_first(ct_dir.iterdir(), lambda p: p.is_file() and "roi" in p.name.lower() and ".nii" in p.name.lower())
        image = load_nifti(ct_path).data.astype(np.float32)
        doctor_roi = load_nifti(roi_path).data != 0
        out_path = out_dir / "labeled_bone_edge" / f"{case}_bone_edge_top{len(rows)}.png"
        render_case(image, rows, out_path, doctor_roi=doctor_roi)
        out_paths.append(out_path)
    return out_paths


def render_unlabeled(data_dir: Path, candidates_csv: Path, out_dir: Path, topk: int) -> list[Path]:
    grouped = group_bone_edge_rows(read_rows(candidates_csv), topk)
    case_dirs = {path.name: path for path in discover_case_dirs(data_dir)}
    out_paths = []
    for case, rows in sorted(grouped.items()):
        image = load_dicom_series(choose_60kev_series(case_dirs[case])).image.data.astype(np.float32)
        out_path = out_dir / "unlabeled_bone_edge" / f"{case}_bone_edge_top{len(rows)}.png"
        render_case(image, rows, out_path, doctor_roi=None)
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render bone-edge tendon candidate previews.")
    parser.add_argument("--labeled-data-dir", default="Data/label")
    parser.add_argument("--unlabeled-data-dir", default="Data/unlabel")
    parser.add_argument("--labeled-candidates", default="outputs/2026-07_bone_edge_tendon_probe/labeled/results/per_case_topk.csv")
    parser.add_argument("--unlabeled-candidates", default="outputs/2026-07_bone_edge_tendon_probe/unlabeled/results/per_case_topk.csv")
    parser.add_argument("--output-dir", default="outputs/2026-07_bone_edge_tendon_probe/previews")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    labeled = render_labeled(Path(args.labeled_data_dir), Path(args.labeled_candidates), out_dir, args.topk)
    unlabeled = render_unlabeled(Path(args.unlabeled_data_dir), Path(args.unlabeled_candidates), out_dir, args.topk)
    make_contact_sheet(labeled, out_dir / "labeled_bone_edge_contact_sheet.png")
    make_contact_sheet(unlabeled, out_dir / "unlabeled_bone_edge_contact_sheet.png")
    print(f"wrote {len(labeled)} labeled and {len(unlabeled)} unlabeled bone-edge previews to {out_dir}")


if __name__ == "__main__":
    main()
