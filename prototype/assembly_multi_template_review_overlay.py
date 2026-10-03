"""Multi-template UI with visible red overlays for already selected templates."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel, QMessageBox, QScrollArea, QVBoxLayout, QWidget

import assembly_multi_template_review_color as colour_app
from dimm_review_app_fixed import RoiCanvas


class TemplateOverlayCanvas(RoiCanvas):
    def __init__(self, image: np.ndarray, max_width: int, max_height: int, existing_rois: list[list[float]]) -> None:
        super().__init__(image, max_width, max_height)
        self.existing_rois = existing_rois

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(235, 50, 50, 60))
        painter.setPen(QPen(QColor(230, 25, 25), 3))
        for index, roi in enumerate(self.existing_rois, 1):
            left = round(roi[0] * self.display_width)
            top = round(roi[1] * self.display_height)
            right = round(roi[2] * self.display_width)
            bottom = round(roi[3] * self.display_height)
            rect = QRect(left, top, max(1, right - left), max(1, bottom - top))
            painter.drawRect(rect)
            painter.fillRect(rect, QColor(235, 50, 50, 60))
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawText(left + 8, top + 28, str(index))
            painter.setPen(QPen(QColor(230, 25, 25), 3))
        painter.end()


class OverlayTemplatePicker(QDialog):
    def __init__(self, image: np.ndarray, existing_rois: list[list[float]], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加稳定定位模板")
        self.setModal(True)
        screen = QApplication.primaryScreen().availableGeometry()
        self.canvas = TemplateOverlayCanvas(image, max(700, screen.width() - 160), max(500, screen.height() - 230), existing_rois)
        hint = QLabel("红色半透明框是已保存的定位模板。请在另一个分散、固定的外壳/文字/螺丝孔区域拖动框选；不要包含插头、线缆或会变化的部件。")
        hint.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确认添加")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        self.resize(min(self.canvas.display_width + 70, screen.width() - 160), min(self.canvas.display_height + 150, screen.height() - 230))

    def confirm(self) -> None:
        if self.canvas.roi() is None:
            QMessageBox.warning(self, "尚未框选", "请先拖动鼠标框选一个足够大的固定区域。")
            return
        self.accept()

    def roi(self) -> list[float] | None:
        return self.canvas.roi()


class OverlayMultiTemplateReview(colour_app.app.MultiTemplateReview):
    def set_template(self) -> None:
        image = self.reference_image()
        if image is None:
            return
        templates = self.recipe.setdefault("template_rois", [])
        if len(templates) >= 3:
            QMessageBox.information(self, "模板数量已足够", "已保存 3 个定位模板；如需重设，请先撤回或重置。")
            return
        picker = OverlayTemplatePicker(image, templates, self)
        if picker.exec_() == QDialog.Accepted:
            templates.append(picker.roi())
            self.save_recipe()
            self.refresh()


colour_app.app.MultiTemplateReview = OverlayMultiTemplateReview


if __name__ == "__main__":
    colour_app.app.main()
