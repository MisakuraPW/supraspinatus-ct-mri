from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NIFTI_DTYPES: dict[int, Any] = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}

NP_TO_NIFTI: dict[Any, tuple[int, int]] = {
    np.dtype("uint8"): (2, 8),
    np.dtype("int16"): (4, 16),
    np.dtype("int32"): (8, 32),
    np.dtype("float32"): (16, 32),
    np.dtype("float64"): (64, 64),
    np.dtype("int8"): (256, 8),
    np.dtype("uint16"): (512, 16),
    np.dtype("uint32"): (768, 32),
}


@dataclass
class NiftiImage:
    data: np.ndarray
    spacing: tuple[float, ...]
    affine: np.ndarray
    header: bytes
    path: Path | None = None


def _open_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def load_nifti(path: str | Path) -> NiftiImage:
    """Load a simple NIfTI-1 image without external dependencies.

    This lightweight reader is enough for the current LHY `.nii.gz` files. For
    larger cloud experiments, nibabel/SimpleITK are still recommended.
    """

    path = Path(path)
    raw = _open_bytes(path)
    header = raw[:352]
    sizeof_hdr_le = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr_le == 348:
        endian = "<"
    elif struct.unpack(">i", header[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"{path} is not a NIfTI-1 file")

    dim = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dim[0])
    shape = tuple(int(v) for v in dim[1 : ndim + 1])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    raw_dtype = NIFTI_DTYPES.get(datatype)
    if raw_dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    dtype = np.dtype(raw_dtype)

    pixdim = struct.unpack(endian + "8f", header[76:108])
    spacing = tuple(float(v) for v in pixdim[1 : ndim + 1])
    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
    slope = struct.unpack(endian + "f", header[112:116])[0]
    inter = struct.unpack(endian + "f", header[116:120])[0]

    count = int(np.prod(shape))
    data = np.frombuffer(raw[vox_offset:], dtype=dtype, count=count).reshape(shape, order="F")
    if slope not in (0.0, 1.0):
        data = data.astype(np.float32) * slope + inter

    affine = np.eye(4, dtype=np.float32)
    for i, sp in enumerate(spacing[:3]):
        affine[i, i] = sp
    return NiftiImage(data=np.asarray(data), spacing=spacing, affine=affine, header=header, path=path)


def save_nifti_like(
    path: str | Path,
    data: np.ndarray,
    reference: NiftiImage | None = None,
    spacing: tuple[float, ...] | None = None,
) -> None:
    """Save a simple NIfTI-1 file, preserving geometry from a reference when possible."""

    path = Path(path)
    arr = np.ascontiguousarray(data)
    dtype = np.dtype(arr.dtype)
    if dtype not in NP_TO_NIFTI:
        arr = arr.astype(np.float32)
        dtype = np.dtype("float32")
    datatype, bitpix = NP_TO_NIFTI[dtype]

    if reference is not None:
        header = bytearray(reference.header[:348])
        spacing = reference.spacing
    else:
        header = bytearray(348)
        struct.pack_into("<i", header, 0, 348)
        spacing = spacing or tuple([1.0] * arr.ndim)

    ndim = arr.ndim
    dims = [ndim] + list(arr.shape) + [1] * (7 - ndim)
    pix = [0.0] + list(spacing or tuple([1.0] * ndim)) + [1.0] * (7 - ndim)
    struct.pack_into("<8h", header, 40, *dims[:8])
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<8f", header, 76, *pix[:8])
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    header[344:348] = b"n+1\0"
    payload = bytes(header) + b"\0\0\0\0" + np.asarray(arr, dtype=dtype).ravel(order="F").tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if str(path).endswith(".gz"):
        with gzip.open(tmp_path, "wb") as f:
            f.write(payload)
    else:
        tmp_path.write_bytes(payload)
    os.replace(tmp_path, path)
