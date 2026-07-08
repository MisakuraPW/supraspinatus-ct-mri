from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from supraspinatus_locator.data.nifti_io import load_nifti
from supraspinatus_locator.data.transforms import normalize_minmax


class NiftiROIDataset(Dataset):
    def __init__(self, pairs: list[tuple[str | Path, str | Path]], clip: tuple[float, float] = (-200.0, 500.0)):
        self.pairs = [(Path(a), Path(b)) for a, b in pairs]
        self.clip = clip

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = load_nifti(image_path).data.astype(np.float32)
        mask = load_nifti(mask_path).data.astype(np.float32)
        image = normalize_minmax(image, self.clip[0], self.clip[1])
        return {
            "image": torch.from_numpy(image[None]),
            "mask": torch.from_numpy((mask > 0).astype(np.float32)[None]),
        }

