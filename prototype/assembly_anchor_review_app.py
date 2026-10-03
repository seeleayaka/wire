"""General assembly visual review with manual four-point alignment.

The operator adds four corresponding, stable anchors on the verified-good
reference and the inspection image in exactly the same order.  The tool then
uses a perspective transform and highlights visual changes for human review.
It never emits an automatic pass/fail decision.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from dimm_upper_right_review import read_image, write_image


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "anchor_reviews"


class ZoomImageView(QGraphicsView):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.item: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.darkGray)
        self.setToolTip("滚轮缩放；按住左键拖动平移；双击恢复适配窗口")

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
        if 0.04 <= self.transform().m11() * factor <= 20:
            self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.fit_image()
        event.accept()


class AnchorCanvas(QLabel):
    def __init__(self, image: np.ndarray, max_width: int, max_height: int) -> None:
        super().__init__()
        self.source_height, self.source_width = image.shape[:2]
        scale = min(1.0, max_width / self.source_width, max_height / self.source_height)
        self.display_width = max(1, round(self.source_width * scale))
        self.display_height = max(1, round(self.source_height * scale))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, self.source_width, self.source_height, self.source_width * 3, QImage.Format_RGB888).copy()
        self.base_pixmap = QPixmap.fromImage(qimage).scaled(self.display_width, self.display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setFixedSize(self.display_width, self.display_height)
        self.setCursor(Qt.CrossCursor)
        self.points: list[QPoint] = []
        self.add_mode = False
        self.render()

    def render(self) -> None:
        pixmap = QPixmap(self.base_pixmap)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.red, 3))
        painter.setBrush(Qt.red)
        for index, point in enumerate(self.points, start=1):
            painter.drawEllipse(point, 8, 8)
            painter.setPen(QPen(Qt.white, 3))
            painter.drawText(point + QPoint(13, -10), str(index))
            painter.setPen(QPen(Qt.red, 3))
        painter.end()
        self.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton and self.add_mode and len(self.points) < 4:
            point = event.pos()
            if self.rect().contains(point):
                self.points.append(point)
                self.add_mode = False
                self.render()
                self.parent().anchor_added()  # type: ignore[union-attr]

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.render()

    def reset(self) -> None:
        self.points.clear()
        self.add_mode = False
        self.render()

    def source_points(self) -> list[list[float]]:
        return [[round(point.x() * self.source_width / self.display_width, 2), round(point.y() * self.source_height / self.display_height, 2)] for point in self.points]


class AnchorPicker(QDialog):
    def __init__(self, image: np.ndarray, role: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"添加四个锚点｜{role}")
        self.setModal(True)
        available = QApplication.primaryScreen().availableGeometry()
        max_width = max(700, available.width() - 160)
        max_height = max(500, available.height() - 250)
        self.canvas = AnchorCanvas(image, max_width, max_height)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.add_button = QPushButton("添加锚点")
        self.undo_button = QPushButton("撤回上一个")
        self.reset_button = QPushButton("重置")
        self.add_button.clicked.connect(self.begin_add)
        self.undo_button.clicked.connect(self.undo)
        self.reset_button.clicked.connect(self.reset)
        controls = QHBoxLayout()
        controls.addWidget(self.add_button)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.reset_button)
        controls.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确认四个锚点")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.hint)
        layout.addLayout(controls)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        self.refresh_hint()
        self.resize(min(self.canvas.display_width + 70, max_width), min(self.canvas.display_height + 190, max_height))

    def refresh_hint(self) -> None:
        number = len(self.canvas.points) + 1
        if len(self.canvas.points) == 4:
            text = "已添加 4/4 个锚点。请确认：两张图片中编号 1、2、3、4 必须是完全对应的刚性位置。"
        elif self.canvas.add_mode:
            text = f"请点击图片添加第 {number} 个锚点。优先选外壳边角、插孔角、固定印字等刚性位置；不要选线缆、插头或会移动的物体。"
        else:
            text = f"当前 {len(self.canvas.points)}/4 个锚点。点击“添加锚点”，再在图片上点击第 {number} 个位置。"
        self.hint.setText(text)
        self.add_button.setEnabled(len(self.canvas.points) < 4)

    def begin_add(self) -> None:
        if len(self.canvas.points) < 4:
            self.canvas.add_mode = True
            self.refresh_hint()

    def anchor_added(self) -> None:
        self.refresh_hint()

    def undo(self) -> None:
        self.canvas.undo()
        self.refresh_hint()

    def reset(self) -> None:
        self.canvas.reset()
        self.refresh_hint()

    def confirm(self) -> None:
        if len(self.canvas.points) != 4:
            QMessageBox.warning(self, "锚点数量不足", "必须添加恰好 4 个锚点后才能对齐。")
            return
        self.accept()

    def points(self) -> list[list[float]]:
        return self.canvas.source_points()


def pick_anchors(image: np.ndarray, role: str, parent: QWidget) -> list[list[float]] | None:
    dialog = AnchorPicker(image, role, parent)
    return dialog.points() if dialog.exec_() == QDialog.Accepted else None


def visual_difference(reference: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]], float]:
    ref_gray = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    target_gray = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX)
    target_gray = cv2.normalize(target_gray, None, 0, 255, cv2.NORM_MINMAX)
    diff = cv2.absdiff(ref_gray, target_gray)
    threshold = max(28.0, float(np.percentile(diff, 99.4)))
    mask = np.uint8(diff >= threshold) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum = max(180, int(mask.shape[0] * mask.shape[1] * 0.00025))
    regions: list[dict[str, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append({"left": float(x), "top": float(y), "right": float(x + width), "bottom": float(y + height), "area": round(area, 1), "difference_score": round(float(diff[y:y + height, x:x + width].mean()), 2)})
    regions.sort(key=lambda item: item["area"] * item["difference_score"], reverse=True)
    return diff, regions, threshold


class AnchorReviewWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI 装配视觉复核｜人工四锚点对齐")
        self.resize(1460, 900)
        self.build()

    def build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        left = QWidget()
        left.setMinimumWidth(370)
        left.setMaximumWidth(480)
        controls = QVBoxLayout(left)
        hint = QLabel("适用范围：插头、DIMM、线缆、卡扣等。\n每次检测先在参考图和待检图按相同顺序添加 4 个刚性锚点；之后才做差异红框。")
        hint.setWordWrap(True)
        hint.setStyleSheet("background:#eef6ff; padding:12px; border-radius:6px")
        controls.addWidget(hint)
        files = QGroupBox("图片")
        form = QFormLayout(files)
        self.reference = QLineEdit()
        self.inspection = QLineEdit()
        for label, edit, handler in (("正确参考图", self.reference, self.choose_reference), ("待检图片", self.inspection, self.choose_inspection)):
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            choose = QPushButton("选择")
            choose.clicked.connect(handler)
            box.addWidget(edit)
            box.addWidget(choose)
            form.addRow(label, row)
        controls.addWidget(files)
        run = QPushButton("添加锚点并开始人工复核")
        run.setMinimumHeight(50)
        run.setStyleSheet("font-size:16px; font-weight:bold; background:#1677ff; color:white")
        run.clicked.connect(self.run)
        controls.addWidget(run)
        self.status = QLabel("选择一张正确参考图和一张待检图。")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addStretch()
        layout.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()
        self.views = [ZoomImageView("尚未检测") for _ in range(3)]
        for view, title in zip(self.views, ("四锚点对齐图", "异常红框", "差异热图")):
            self.tabs.addTab(view, title)
        right_layout.addWidget(self.tabs, 3)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("检测完成后显示锚点、对齐和人工复核报告。")
        right_layout.addWidget(self.report, 1)
        layout.addWidget(right, 1)

    def choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择正确参考图", str(Path.home() / "Desktop"), "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.reference.setText(path)

    def choose_inspection(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择待检图", str(Path.home() / "Desktop"), "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.inspection.setText(path)

    def run(self) -> None:
        reference_path = Path(self.reference.text().strip())
        inspection_path = Path(self.inspection.text().strip())
        if not reference_path.is_file() or not inspection_path.is_file():
            QMessageBox.warning(self, "缺少图片", "请选择有效的正确参考图和待检图。")
            return
        try:
            reference = read_image(reference_path)
            inspection = read_image(inspection_path)
        except Exception as error:
            QMessageBox.critical(self, "读取失败", str(error))
            return
        reference_points = pick_anchors(reference, "第一步：正确参考图", self)
        if reference_points is None:
            self.status.setText("已取消参考图锚点添加。")
            return
        inspection_points = pick_anchors(inspection, "第二步：待检图（必须按参考图的 1→4 相同顺序）", self)
        if inspection_points is None:
            self.status.setText("已取消待检图锚点添加。")
            return
        transform = cv2.getPerspectiveTransform(np.float32(inspection_points), np.float32(reference_points))
        height, width = reference.shape[:2]
        aligned = cv2.warpPerspective(inspection, transform, (width, height))
        difference, regions, threshold = visual_difference(reference, aligned)
        overlay = aligned.copy()
        for index, region in enumerate(regions, start=1):
            left, top, right, bottom = (int(region[key]) for key in ("left", "top", "right", "bottom"))
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 4)
            cv2.putText(overlay, str(index), (left, max(35, top - 9)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        output.mkdir(parents=True, exist_ok=True)
        write_image(output / "aligned.jpg", aligned)
        write_image(output / "anomaly_boxes.jpg", overlay)
        write_image(output / "difference_heatmap.jpg", cv2.applyColorMap(difference, cv2.COLORMAP_JET))
        report = {"prototype": True, "decision": "manual_review_required", "alignment": {"method": "manual_four_point_perspective", "reference_anchor_points": reference_points, "inspection_anchor_points": inspection_points}, "reference": str(reference_path), "inspection": str(inspection_path), "difference_threshold": round(threshold, 2), "review_region_count": len(regions), "review_regions": regions, "note": "红框表示视觉变化区域。四个锚点必须选在不会移动的刚性主体上；不要选线缆、插头或阴影。"}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.views[0].load(output / "aligned.jpg")
        self.views[1].load(output / "anomaly_boxes.jpg")
        self.views[2].load(output / "difference_heatmap.jpg")
        self.tabs.setCurrentIndex(1)
        self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        self.status.setText(f"已用人工四锚点对齐。发现 {len(regions)} 个红框复核区；结果已保存到：{output}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AnchorReviewWindow()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
