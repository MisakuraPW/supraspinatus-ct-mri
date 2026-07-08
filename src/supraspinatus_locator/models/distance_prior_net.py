from __future__ import annotations

from .roi_unet3d import ROIUNet3D


class DistancePriorNet(ROIUNet3D):
    """ROI network expecting CT plus prior channels such as bone mask/distance maps/coordinates."""

    def __init__(self, prior_channels: int = 4, out_channels: int = 1, base_channels: int = 16):
        super().__init__(in_channels=1 + prior_channels, out_channels=out_channels, base_channels=base_channels)

