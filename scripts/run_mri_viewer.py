from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from _bootstrap import add_src_to_path

add_src_to_path()

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.viewer.volume_viewer import ViewerData, run_viewer


@dataclass
class MRSeries:
    case: str
    path: Path
    files: list[Path]
    metadata: dict[str, Any]


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def dicom_sort_key(path: Path) -> tuple[float, int, str]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return (0.0, 0, path.name)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return (float(ipp[2]), int(getattr(ds, "InstanceNumber", 0) or 0), path.name)
        except Exception:
            pass
    return (float(int(getattr(ds, "InstanceNumber", 0) or 0)), int(getattr(ds, "InstanceNumber", 0) or 0), path.name)


def read_dicom_header(path: Path) -> Any | None:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    if not getattr(ds, "Modality", None):
        return None
    return ds


def infer_case_name(series_path: Path, data_root: Path) -> str:
    try:
        rel = series_path.relative_to(data_root)
    except ValueError:
        return series_path.name
    parts = rel.parts
    if len(parts) >= 2 and parts[0].lower() in {"label", "labeled", "unlabel", "unlabeled"}:
        return parts[1]
    if len(parts) >= 1:
        return parts[0]
    return series_path.name


def discover_mr_series(root: str | Path) -> list[MRSeries]:
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"MR search root is not a directory: {root}")
    search_roots: list[Path] = []
    if root.name.upper() == "MR" or any(read_dicom_header(path) is not None for path in list(root.iterdir())[:3] if path.is_file()):
        search_roots.append(root)
    if (root / "MR").is_dir():
        search_roots.append(root / "MR")
    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir() and (child / "MR").is_dir():
            search_roots.append(child / "MR")
        if child.is_dir():
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "MR").is_dir():
                    search_roots.append(grandchild / "MR")
    if not search_roots:
        search_roots = [root]

    dicom_files: list[tuple[Path, Any]] = []
    for search_root in dict.fromkeys(search_roots):
        for path in search_root.rglob("*"):
            if not path.is_file() or is_nifti(path):
                continue
            ds = read_dicom_header(path)
            if ds is None:
                continue
            modality = str(getattr(ds, "Modality", "")).upper()
            if modality not in {"MR", "MRI"}:
                continue
            dicom_files.append((path, ds))

    grouped: dict[str, list[tuple[Path, Any]]] = {}
    for path, ds in dicom_files:
        uid = str(getattr(ds, "SeriesInstanceUID", path.parent.as_posix()))
        grouped.setdefault(uid, []).append((path, ds))

    series_list: list[MRSeries] = []
    for uid, items in grouped.items():
        files = sorted((path for path, _ds in items), key=dicom_sort_key)
        first = items[0][1]
        parent = files[0].parent
        rows = int(getattr(first, "Rows", 0) or 0)
        cols = int(getattr(first, "Columns", 0) or 0)
        ps = getattr(first, "PixelSpacing", [1.0, 1.0])
        try:
            spacing_xy = [float(ps[0]), float(ps[1])]
        except Exception:
            spacing_xy = [1.0, 1.0]
        metadata = {
            "series_uid": uid,
            "modality": str(getattr(first, "Modality", "")),
            "series_description": str(getattr(first, "SeriesDescription", "")),
            "protocol_name": str(getattr(first, "ProtocolName", "")),
            "sequence_name": str(getattr(first, "SequenceName", "")),
            "patient_name": str(getattr(first, "PatientName", "")),
            "rows": rows,
            "columns": cols,
            "count": len(files),
            "pixel_spacing": spacing_xy,
            "slice_thickness": float(getattr(first, "SliceThickness", 0.0) or 0.0),
        }
        series_list.append(MRSeries(case=infer_case_name(parent, root), path=parent, files=files, metadata=metadata))
    series_list.sort(key=lambda item: (item.case, item.path.as_posix(), item.metadata["series_description"]))
    return series_list


def load_dicom_series(series: MRSeries) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    slices = []
    z_positions = []
    for path in series.files:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        slices.append(arr)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            try:
                z_positions.append(float(ipp[2]))
            except Exception:
                pass
    volume = np.stack(slices, axis=-1)
    ps = series.metadata.get("pixel_spacing") or [1.0, 1.0]
    if len(z_positions) > 1:
        unique = sorted(set(z_positions))
        diffs = np.diff(unique)
        z_spacing = float(np.median(np.abs(diffs))) if len(diffs) else float(series.metadata.get("slice_thickness") or 1.0)
    else:
        z_spacing = float(series.metadata.get("slice_thickness") or 1.0)
    return volume, (float(ps[0]), float(ps[1]), z_spacing), series.metadata


def find_auto_mask(image_path: Path) -> Path | None:
    search_roots = []
    if image_path.is_dir():
        search_roots.extend([image_path, image_path.parent])
    else:
        search_roots.extend([image_path.parent, image_path.parent.parent])
    patterns = ("*roi*.nii*", "*ROI*.nii*", "*label*.nii*", "*mask*.nii*")
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(path for path in root.rglob(pattern) if path.is_file() and is_nifti(path))
            if matches:
                return matches[0]
    return None


def choose_series(series_list: list[MRSeries], case: str | None, series_index: int | None, contains: str | None) -> MRSeries:
    pool = series_list
    if case:
        pool = [item for item in pool if item.case == case or case.lower() in item.case.lower()]
    if contains:
        q = contains.lower()
        pool = [
            item
            for item in pool
            if q in str(item.metadata.get("series_description", "")).lower()
            or q in str(item.metadata.get("protocol_name", "")).lower()
            or q in item.path.as_posix().lower()
        ]
    if not pool:
        raise SystemExit("No matching MR series found. Run with --list to inspect available series.")
    if series_index is not None:
        if series_index < 0 or series_index >= len(pool):
            raise SystemExit(f"--series-index must be in [0, {len(pool) - 1}] after filtering.")
        return pool[series_index]
    scored = sorted(
        pool,
        key=lambda item: (
            -int("cor" in str(item.metadata.get("series_description", "")).lower()),
            -int("t2" in str(item.metadata.get("series_description", "")).lower()),
            -int("fs" in str(item.metadata.get("series_description", "")).lower() or "fat" in str(item.metadata.get("series_description", "")).lower()),
            -int(item.metadata.get("count", 0)),
        ),
    )
    return scored[0]


def auto_window_params(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    width = max(float(hi - lo), 1e-6)
    center = float((hi + lo) / 2.0)
    return center, width


def load_overlay_if_compatible(path: Path | str, image_shape: tuple[int, ...], label: str) -> np.ndarray | None:
    overlay = load_nifti(path).data
    if overlay.shape != image_shape:
        print(f"Skip {label}: shape {overlay.shape} does not match image shape {image_shape}.")
        return None
    return overlay


def print_series_table(series_list: list[MRSeries], csv_path: Path | None = None) -> None:
    rows = []
    for idx, item in enumerate(series_list):
        row = {
            "index": idx,
            "case": item.case,
            "count": item.metadata.get("count", ""),
            "shape": f"{item.metadata.get('columns', '')}x{item.metadata.get('rows', '')}x{item.metadata.get('count', '')}",
            "description": item.metadata.get("series_description", ""),
            "protocol": item.metadata.get("protocol_name", ""),
            "path": str(item.path),
        }
        rows.append(row)
        print(
            f"[{idx:02d}] case={row['case']} shape={row['shape']} "
            f"desc={row['description']} protocol={row['protocol']} path={row['path']}"
        )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index", "case", "count", "shape", "description", "protocol", "path"])
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open shoulder MRI DICOM/NIfTI in the interactive viewer.")
    parser.add_argument("--data-root", default="Data/label", help="Root used for --list and case/series discovery.")
    parser.add_argument("--image", help="NIfTI image or DICOM series directory. If omitted, use discovered MR series.")
    parser.add_argument("--case", help="Case name filter, e.g. LHY or HMC.")
    parser.add_argument("--series-index", type=int, help="Index after applying --case/--contains filters.")
    parser.add_argument("--contains", help="Filter MR series by description/protocol/path substring, e.g. cor or T2.")
    parser.add_argument("--mask", help="Optional NIfTI mask/ROI overlay. Use 'auto' to search near the image.")
    parser.add_argument("--pred", help="Optional NIfTI prediction overlay.")
    parser.add_argument("--list", action="store_true", help="List discovered MR DICOM series and exit.")
    parser.add_argument("--list-csv", default="outputs/mri_series_index.csv", help="CSV path for --list output.")
    parser.add_argument("--window-center", type=float, help="Display window center. Defaults to MRI percentile auto window.")
    parser.add_argument("--window-width", type=float, help="Display window width. Defaults to MRI percentile auto window.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.list:
        series_list = discover_mr_series(data_root)
        print_series_table(series_list, Path(args.list_csv) if args.list_csv else None)
        return

    mask = None
    pred = None
    image_path: Path
    if args.image:
        image_path = Path(args.image)
        if image_path.is_file() and is_nifti(image_path):
            nifti = load_nifti(image_path)
            image = nifti.data.astype(np.float32)
        elif image_path.is_dir():
            series_list = discover_mr_series(image_path)
            series = choose_series(series_list, None, args.series_index, args.contains)
            image, _spacing, meta = load_dicom_series(series)
            print(f"Opening MR series: case={series.case} desc={meta.get('series_description', '')} path={series.path}")
        else:
            raise SystemExit(f"Unsupported --image path: {image_path}")
    else:
        series_list = discover_mr_series(data_root)
        series = choose_series(series_list, args.case, args.series_index, args.contains)
        image_path = series.path
        image, _spacing, meta = load_dicom_series(series)
        print(f"Opening MR series: case={series.case} desc={meta.get('series_description', '')} path={series.path}")

    if args.mask:
        mask_path = find_auto_mask(image_path) if args.mask.lower() == "auto" else Path(args.mask)
        if mask_path is None:
            print("No auto mask found.")
        else:
            print(f"Overlay mask: {mask_path}")
            mask = load_overlay_if_compatible(mask_path, image.shape, "mask")
    if args.pred:
        pred = load_overlay_if_compatible(args.pred, image.shape, "pred")

    if args.window_center is None or args.window_width is None:
        center, width = auto_window_params(image)
        if args.window_center is not None:
            center = args.window_center
        if args.window_width is not None:
            width = args.window_width
    else:
        center, width = args.window_center, args.window_width
    raise SystemExit(run_viewer(ViewerData(image=image, mask=mask, pred=pred, window_center=center, window_width=width)))


if __name__ == "__main__":
    main()
