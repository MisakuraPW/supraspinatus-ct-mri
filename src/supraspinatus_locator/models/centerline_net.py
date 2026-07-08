from __future__ import annotations

from .roi_unet3d import ROIUNet3D


class CenterlineNet(ROIUNet3D):
    """Predicts a tendon corridor or centerline heatmap instead of a hard mask."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__(in_channels=in_channels, out_channels=1, base_channels=base_channels)

