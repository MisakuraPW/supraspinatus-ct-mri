from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from supraspinatus_locator.data.nifti_io import load_nifti, save_nifti_like
from supraspinatus_locator.data.transforms import normalize_minmax
from supraspinatus_locator.models.roi_unet3d import ROIUNet3D


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROI U-Net inference on one NIfTI volume.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    img = load_nifti(args.image)
    x = normalize_minmax(img.data.astype(np.float32), -200.0, 500.0)
    tensor = torch.from_numpy(x[None, None]).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ROIUNet3D().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor.to(device))).cpu().numpy()[0, 0]
    mask = (prob >= args.threshold).astype(np.uint8)
    save_nifti_like(Path(args.out), mask, reference=img)


if __name__ == "__main__":
    main()

