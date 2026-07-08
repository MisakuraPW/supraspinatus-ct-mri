from __future__ import annotations

import numpy as np


def color_overlay(gray: np.ndarray, overlays: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
    for mask, color, alpha in overlays:
        m = mask > 0
        c = np.asarray(color, dtype=np.float32)
        rgb[m] = rgb[m] * (1.0 - alpha) + c * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)

