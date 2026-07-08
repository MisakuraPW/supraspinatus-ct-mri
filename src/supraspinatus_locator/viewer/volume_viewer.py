from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from supraspinatus_locator.preprocessing.ct_windowing import apply_window

from .overlay import color_overlay


@dataclass
class ViewerData:
    image: np.ndarray
    mask: np.ndarray | None = None
    pred: np.ndarray | None = None
    window_center: float = 80.0
    window_width: float = 500.0


class VolumeViewer(QtWidgets.QMainWindow):
    def __init__(self, data: ViewerData):
        super().__init__()
        self.data = data
        self.axis = 2
        self.index = data.image.shape[self.axis] // 2
        self.setWindowTitle("Supraspinatus ROI Viewer")

        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.set_index)
        self.axis_box = QtWidgets.QComboBox()
        self.axis_box.addItems(["Axial/Z", "Coronal/Y", "Sagittal/X"])
        self.axis_box.currentIndexChanged.connect(self.set_axis)
        self.info = QtWidgets.QLabel()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.axis_box)
        controls.addWidget(self.slider)
        controls.addWidget(self.info)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(controls)
        root = QtWidgets.QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self._reset_slider()
        self.render_slice()

    def _reset_slider(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.data.image.shape[self.axis] - 1)
        self.index = self.data.image.shape[self.axis] // 2
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)

    def set_axis(self, idx: int) -> None:
        self.axis = {0: 2, 1: 1, 2: 0}[idx]
        self._reset_slider()
        self.render_slice()

    def set_index(self, value: int) -> None:
        self.index = int(value)
        self.render_slice()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = 1 if event.angleDelta().y() > 0 else -1
        self.slider.setValue(np.clip(self.index + delta, self.slider.minimum(), self.slider.maximum()))

    def _slice(self, arr: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            return arr[self.index, :, :]
        if self.axis == 1:
            return arr[:, self.index, :]
        return arr[:, :, self.index]

    def render_slice(self) -> None:
        win = apply_window(self.data.image, self.data.window_center, self.data.window_width)
        gray = np.rot90(self._slice(win))
        overlays = []
        if self.data.mask is not None:
            overlays.append((np.rot90(self._slice(self.data.mask)), (255, 64, 64), 0.35))
        if self.data.pred is not None:
            overlays.append((np.rot90(self._slice(self.data.pred)), (64, 220, 255), 0.30))
        rgb = color_overlay(gray, overlays)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)
        self.info.setText(f"axis={self.axis} slice={self.index}/{self.data.image.shape[self.axis] - 1}")


def run_viewer(data: ViewerData) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = VolumeViewer(data)
    viewer.resize(900, 760)
    viewer.show()
    return app.exec_()

