"""Zoomable/pannable launcher for the fixed DIMM review UI."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

import dimm_review_app as base
from dimm_review_app_fixed import label_roi


class ZoomImageView(QGraphicsView):
    """Result canvas: wheel zooms at cursor; left-drag pans the image."""
    def __init__(self, message: str) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.item: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(Qt.darkGray)
        self.setToolTip("滚轮：以光标位置缩放；按住左键拖动：移动图片；双击：恢复适配窗口")

    def load(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        self.scene.clear()
        self.item = None
        if pixmap.isNull():
            self.scene.addText(f"无法载入：{path}")
            return
        self.item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(self.item.boundingRect())
        self.resetTransform()
        self.fit_image()

    def fit_image(self) -> None:
        if self.item is not None:
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.item is None:
            event.ignore()
            return
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        current = self.transform().m11()
        if 0.04 <= current * factor <= 20:
            self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.fit_image()
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self.item is not None and self.transform().m11() == 1.0:
            self.fit_image()


def main() -> None:
    # Reuse the reviewed detector and ROI picker, replacing only result viewing.
    base.label_roi = label_roi
    base.ImageView = ZoomImageView
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = base.Window()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
