from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from supraspinatus_locator.preprocessing.ct_windowing import apply_window


def save_overlay_montage(
    image: np.ndarray,
    out_path: str | Path,
    masks: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
    center: int | None = None,
    window_center: float = 80.0,
    window_width: float = 500.0,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = image.shape[2] // 2 if center is None else int(center)
    indices = [max(0, z - 6), z, min(image.shape[2] - 1, z + 6)]
    tiles = []
    win = apply_window(image, window_center, window_width)
    for idx in indices:
        gray = win[:, :, idx]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if masks:
            for mask, color in masks:
                overlay = np.zeros_like(rgb)
                overlay[mask[:, :, idx] > 0] = color
                rgb = cv2.addWeighted(rgb, 1.0, overlay, 0.35, 0)
        cv2.putText(rgb, f"slice {idx}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(rgb)
    montage = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(out_path), montage)

