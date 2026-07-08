from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from supraspinatus_locator.models.roi_unet3d import ROIUNet3D
from supraspinatus_locator.training.datasets import NiftiROIDataset
from supraspinatus_locator.training.losses import DiceBCELoss


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image, mask = line.split(",", 1)
        pairs.append((image.strip(), mask.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple 3D ROI U-Net.")
    parser.add_argument("--pairs", required=True, help="CSV-like txt: image_path,mask_path per line")
    parser.add_argument("--out", default="checkpoints/roi_unet3d.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    dataset = NiftiROIDataset(read_pairs(Path(args.pairs)))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ROIUNet3D().to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(image), mask)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        print(f"epoch={epoch + 1} loss={total / max(len(loader), 1):.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)


if __name__ == "__main__":
    main()

