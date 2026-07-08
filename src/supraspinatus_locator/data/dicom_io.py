from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


@dataclass
class DicomSeries:
    path: Path
    files: list[Path]
    metadata: dict[str, Any]


def scan_dicom_series(path: str | Path) -> DicomSeries:
    root = Path(path)
    files = [p for p in root.iterdir() if p.is_file()]
    dicoms: list[tuple[Path, Any]] = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", None):
                dicoms.append((f, ds))
        except Exception:
            continue
    if not dicoms:
        raise ValueError(f"No DICOM files found under {root}")

    first = dicoms[0][1]
    metadata = {
        "modality": str(getattr(first, "Modality", "")),
        "series_description": str(getattr(first, "SeriesDescription", "")),
        "protocol_name": str(getattr(first, "ProtocolName", "")),
        "rows": int(getattr(first, "Rows", 0)),
        "columns": int(getattr(first, "Columns", 0)),
        "pixel_spacing": [float(x) for x in getattr(first, "PixelSpacing", [])],
        "slice_thickness": float(getattr(first, "SliceThickness", 0.0) or 0.0),
        "rescale_slope": float(getattr(first, "RescaleSlope", 1.0) or 1.0),
        "rescale_intercept": float(getattr(first, "RescaleIntercept", 0.0) or 0.0),
        "count": len(dicoms),
    }
    sorted_files = sorted(
        (p for p, _ in dicoms),
        key=lambda p: int(getattr(pydicom.dcmread(str(p), stop_before_pixels=True, force=True), "InstanceNumber", 0)),
    )
    return DicomSeries(path=root, files=sorted_files, metadata=metadata)


def load_dicom_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    series = scan_dicom_series(path)
    slices = []
    z_positions = []
    for f in series.files:
        ds = pydicom.dcmread(str(f), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        slices.append(arr)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            z_positions.append(float(ipp[2]))
    volume = np.stack(slices, axis=-1)
    ps = series.metadata["pixel_spacing"] or [1.0, 1.0]
    if len(z_positions) > 1:
        diffs = np.diff(sorted(set(z_positions)))
        z_spacing = float(np.median(np.abs(diffs))) if len(diffs) else series.metadata["slice_thickness"]
    else:
        z_spacing = series.metadata["slice_thickness"] or 1.0
    return volume, (float(ps[0]), float(ps[1]), z_spacing), series.metadata

