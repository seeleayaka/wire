"""Fixed ROI-picker launcher for dimm_review_app.

It replaces OpenCV's blocking selectROIs window with a resizable PyQt dialog
that has explicit Confirm/Cancel controls and a normal close button.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QMessageBox, QRubberBand, QScrollArea, QVBoxLayout, QWidget

import dimm_review_app as base
from dimm_upper_right_review import read_image


class RoiCanvas(QLabel):
    def __init__(self, image: np.ndarray, max_width: int, max_height: int) -> None:
        super().__init__()
        source_height, source_width = image.shape[:2]
        scale = min(1.0, max_width / source_width, max_height / source_height)
        self.display_width = max(1, round(source_width * scale))
        self.display_height = max(1, round(source_height * scale))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, source_width, source_height, source_width * 3, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qimage).scaled(self.display_width, self.display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setFixedSize(self.display_width, self.display_height)
        self.setCursor(Qt.CrossCursor)
        self.origin: QPoint | None = None
        self.selection: QRect | None = None
        self.band = QRubberBand(QRubberBand.Rectangle, self)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.selection = QRect(self.origin, self.origin)
            self.band.setGeometry(self.selection)
            self.band.show()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.origin is not None:
            self.selection = QRect(self.origin, event.pos()).normalized().intersected(self.rect())
            self.band.setGeometry(self.selection)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton and self.origin is not None:
            self.selection = QRect(self.origin, event.pos()).normalized().intersected(self.rect())
            self.band.setGeometry(self.selection)
            self.origin = None

    def roi(self) -> list[float] | None:
        if self.selection is None or self.selection.width() < 8 or self.selection.height() < 8:
            return None
        rect = self.selection
        return [round(rect.left() / self.display_width, 6), round(rect.top() / self.display_height, 6), round((rect.right() + 1) / self.display_width, 6), round((rect.bottom() + 1) / self.display_height, 6)]


class RoiPicker(QDialog):
    def __init__(self, image: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("框选右上 DIMM 槽组")
        self.setModal(True)
        available = QApplication.primaryScreen().availableGeometry()
        max_width = max(700, available.width() - 160)
        max_height = max(500, available.height() - 230)
        self.canvas = RoiCanvas(image, max_width, max_height)
        hint = QLabel("拖动鼠标框选完整的右上 DIMM 槽组。窗口可拖动边缘调整大小；点击“取消”或右上角 × 会直接返回主界面。")
        hint.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确认保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        self.resize(min(self.canvas.display_width + 70, max_width), min(self.canvas.display_height + 150, max_height))

    def confirm(self) -> None:
        if self.canvas.roi() is None:
            QMessageBox.warning(self, "尚未框选", "请先拖动鼠标框选一个足够大的 DIMM 槽组区域。")
            return
        self.accept()


def label_roi(path: Path) -> list[float] | None:
    dialog = RoiPicker(read_image(path))
    return dialog.canvas.roi() if dialog.exec_() == QDialog.Accepted else None


def main() -> None:
    # Window.mark resolves label_roi in the original module's global namespace.
    base.label_roi = label_roi
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = base.Window()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
