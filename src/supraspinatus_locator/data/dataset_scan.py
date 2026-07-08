from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dicom_io import scan_dicom_series
from .nifti_io import load_nifti


def summarize_nifti(path: Path) -> dict[str, Any]:
    img = load_nifti(path)
    data = img.data
    nonzero = np.argwhere(data != 0)
    bbox = None
    if nonzero.size:
        bbox = {"min": nonzero.min(axis=0).tolist(), "max": nonzero.max(axis=0).tolist()}
    return {
        "path": str(path),
        "kind": "nifti",
        "shape": list(data.shape),
        "spacing": list(img.spacing),
        "dtype": str(data.dtype),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "nonzero_voxels": int(np.count_nonzero(data)),
        "bbox_nonzero": bbox,
    }


def scan_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    result: dict[str, Any] = {"root": str(root), "nifti": [], "dicom_series": []}
    for p in root.rglob("*.nii.gz"):
        result["nifti"].append(summarize_nifti(p))
    for p in root.rglob("*.nii"):
        result["nifti"].append(summarize_nifti(p))
    for d in sorted([p for p in root.rglob("*") if p.is_dir()]):
        files = [x for x in d.iterdir() if x.is_file()]
        if not files:
            continue
        try:
            series = scan_dicom_series(d)
        except Exception:
            continue
        result["dicom_series"].append({"path": str(d), **series.metadata})
    return result


def write_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

