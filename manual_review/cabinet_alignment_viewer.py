"""Small manual viewer for a reference, an inspection image and their alignment.

This is deliberately a viewer, not an inspection decision tool.  It reuses the
current mainline's SIFT + USAC_MAGSAC homography, keeps the inputs read-only,
and lets a person decide whether the registration visually makes sense before
any difference/candidate step is considered.

Run with:
    E:\PythonProject10\.venv\Scripts\python.exe -B manual_review\cabinet_alignment_viewer.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


PROJECT = Path(__file__).resolve().parents[1]
VIEWER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "prototype"))
sys.path.insert(0, str(VIEWER_DIR))

# This import is only the already-audited global registration routine.  It does
# not invoke DINO or generate candidate regions.
import assembly_auto_review_robust_v3 as perspective  # noqa: E402
from local_band_alignment_probe import (  # noqa: E402
    BANDS,
    bounds_from_normalized,
    edge_overlap_f1,
    estimate_local_homography,
    labelled_panel,
    local_to_full_transform,
)


CASE_DIR = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\4")
DEFAULT_REFERENCE = CASE_DIR / "right.png"
DEFAULT_INSPECTION = CASE_DIR / "wrong13.png"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)"


def read_image(path: Path) -> np.ndarray:
    """Unicode-safe BGR image reader for Chinese Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def to_qimage(image: np.ndarray) -> QImage:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("仅支持三通道彩色图像")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, width * channels, QImage.Format_RGB888).copy()


def format_alignment_report(report: dict[str, Any]) -> str:
    quality = report.get("alignment_quality", {})
    reliable = quality.get("reliable", False)
    status = "通过" if reliable else "不确定 / 已拒绝"
    lines = [
        f"对齐状态：{status}",
        "方向：待检图 → 原图坐标（原图本身不变形）",
        f"方法：{report.get('method', 'unknown')}",
        f"参考关键点 / 待检关键点：{report.get('reference_keypoints', '-')} / {report.get('inspection_keypoints', '-')}",
        f"匹配 / 内点：{report.get('matches', '-')} / {report.get('inliers', '-')}",
        f"内点比例：{report.get('inlier_ratio', '-')}",
        f"中位重投影误差：{report.get('median_reprojection_error', '-')} px",
        f"有效对齐区域覆盖：{report.get('valid_warp_coverage', '-')}",
    ]
    if report.get("reason"):
        lines.append(f"说明：{report['reason']}")
    checks = quality.get("checks", {})
    if checks:
        lines.append("\n安全检查：")
        for key in ("inlier_count_ok", "inlier_ratio_ok", "spatial_coverage_ok", "reprojection_error_ok", "area_ratio_ok", "geometry_ok"):
            if key in checks:
                lines.append(f"  {key}: {'通过' if checks[key] else '未通过'}")
    lines.append("\n边界：此窗口只帮助人工看对齐；不输出故障结论、不生成候选框。")
    return "\n".join(lines)


def compute_local_band_results(reference: np.ndarray, aligned: np.ndarray) -> dict[str, dict[str, Any]]:
    """Compute Cabinet-5 band comparisons in memory, without writing or stitching images.

    A local matrix is only used for a displayed crop when its tolerant edge
    overlap is meaningfully better than the existing global result.  This is a
    display-choice heuristic, not a fault score or a validation metric.
    """
    results: dict[str, dict[str, Any]] = {}
    for name, normalized_roi in BANDS:
        x1, y1, x2, y2 = bounds_from_normalized(reference, normalized_roi)
        reference_crop = reference[y1:y2, x1:x2]
        global_crop = aligned[y1:y2, x1:x2]
        local, local_report = estimate_local_homography(reference_crop, global_crop)
        local_crop = global_crop
        if local is not None:
            full_transform = local_to_full_transform(local, (x1, y1, x2, y2))
            local_full = cv2.warpPerspective(
                aligned,
                full_transform,
                (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            local_crop = local_full[y1:y2, x1:x2]
        global_f1 = edge_overlap_f1(reference_crop, global_crop)
        local_f1 = edge_overlap_f1(reference_crop, local_crop)
        delta = None if global_f1 is None or local_f1 is None else round(local_f1 - global_f1, 4)
        # A tiny F1 wobble is not enough to replace the established global H.
        use_local = local is not None and delta is not None and delta > 0.005
        results[name] = {
            "pixel_bounds": (x1, y1, x2, y2),
            "reference_crop": reference_crop,
            "global_crop": global_crop,
            "local_crop": local_crop,
            "local_alignment": local_report,
            "global_edge_overlap_f1": global_f1,
            "local_edge_overlap_f1": local_f1,
            "edge_overlap_delta": delta,
            "use_local_for_manual_preview": use_local,
        }
    return results


def format_local_band_report(results: dict[str, dict[str, Any]]) -> str:
    if not results:
        return "\n局部平面实验：未计算。"
    names = {"upper_device_plane": "上排器件平面", "lower_device_plane": "下排器件平面"}
    lines = [
        "\n局部平面实验（仅供人工查看，不接入主链路）：",
        "每一块都是全局 H 后的独立 3×3 H；不做网格、光流、液化或全图拼接。",
    ]
    for name, band in results.items():
        report = band["local_alignment"]
        choice = "局部 H（该显示块边缘重合度更高）" if band["use_local_for_manual_preview"] else "保留全局 H"
        lines.append(
            f"- {names.get(name, name)}：全局边缘 F1={band['global_edge_overlap_f1']}，"
            f"局部边缘 F1={band['local_edge_overlap_f1']}，变化={band['edge_overlap_delta']}；显示选择：{choice}。"
        )
        lines.append(
            f"  局部匹配 / 内点：{report.get('matches', '-')} / {report.get('inliers', '-')}，"
            f"中位重投影误差：{report.get('median_reprojection_error_pixels', '-')} px，"
            f"局部 H：{report.get('reason', '-')}。"
        )
    lines.append("这只是对齐观感的辅助量，不代表检出率、误报率或故障结论。")
    return "\n".join(lines)


class ZoomableImageView(QGraphicsView):
    """Image view with wheel zoom and hand drag; each panel is independent."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: QGraphicsPixmapItem | None = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.darkGray)
        self.setFrameShape(QFrame.StyledPanel)

    def set_image(self, image: np.ndarray | None) -> None:
        self._scene.clear()
        self._item = None
        if image is not None:
            self._item = self._scene.addPixmap(QPixmap.fromImage(to_qimage(image)))
            self._scene.setSceneRect(self._item.boundingRect())
        self.fit_image()

    def fit_image(self) -> None:
        if self._item is None:
            return
        self.resetTransform()
        self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: Any) -> None:
        if self._item is None:
            event.ignore()
            return
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)


class ImagePanel(QWidget):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600; font-size: 14px;")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("color: #666;")
        self.view = ZoomableImageView(self)
        self.fit_button = QPushButton("适应窗口")
        self.fit_button.clicked.connect(self.view.fit_image)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.fit_button, 0, Qt.AlignRight)

    def set_image(self, image: np.ndarray | None) -> None:
        self.view.set_image(image)


class CabinetAlignmentViewer(QMainWindow):
    def __init__(self, reference_path: Path | None = None, inspection_path: Path | None = None) -> None:
        super().__init__()
        self._reference: np.ndarray | None = None
        self._inspection: np.ndarray | None = None
        self._aligned: np.ndarray | None = None
        self._report: dict[str, Any] | None = None
        self._band_results: dict[str, dict[str, Any]] = {}

        self.setWindowTitle("线材柜对齐查看器（人工复核｜分区实验版）")
        self.resize(1760, 960)
        self._build_ui()
        self.reference_path.setText(str(reference_path or DEFAULT_REFERENCE))
        self.inspection_path.setText(str(inspection_path or DEFAULT_INSPECTION))
        if Path(self.reference_path.text()).is_file() and Path(self.inspection_path.text()).is_file():
            self.run_alignment()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(central)

        controls = QGroupBox("输入与对齐")
        control_layout = QFormLayout(controls)
        self.reference_path = QLineEdit()
        self.inspection_path = QLineEdit()
        ref_row = self._path_row(self.reference_path, "选择原图")
        inspection_row = self._path_row(self.inspection_path, "选择待检图")
        control_layout.addRow("原图（reference）：", ref_row)
        control_layout.addRow("待检图（inspection）：", inspection_row)
        self.align_button = QPushButton("执行 SIFT/MAGSAC 对齐")
        self.align_button.clicked.connect(self.run_alignment)
        self.status = QLabel("请选择两张图片后执行对齐。")
        self.status.setStyleSheet("font-weight: 600; color: #455a64;")
        action_row = QHBoxLayout()
        action_row.addWidget(self.align_button)
        action_row.addWidget(self.status, 1)
        control_layout.addRow(action_row)
        root.addWidget(controls)

        self.reference_panel = ImagePanel("① 原图", "基准图：保持原始像素，不做形变。")
        self.inspection_panel = ImagePanel("② 待检图", "原始待检图：保持原始像素，不做形变。")
        self.result_panel = ImagePanel("③ 对齐结果", "可查看全局 H，或上/下排分别的局部 H 对比。")
        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.reference_panel)
        image_splitter.addWidget(self.inspection_panel)
        image_splitter.addWidget(self.result_panel)
        image_splitter.setSizes([580, 580, 640])

        options = QGroupBox("第三栏显示")
        options_layout = QHBoxLayout(options)
        self.result_mode = QComboBox()
        self.result_mode.addItem("50/50 叠图：原图 + 全局对齐待检图", "global_blend")
        self.result_mode.addItem("仅显示全局对齐后的待检图", "global_aligned")
        self.result_mode.addItem("分区预览（实验）：上/下排各自择优", "band_preview")
        self.result_mode.addItem("上排对比（实验）：原图｜全局 H｜局部 H", "upper_device_plane")
        self.result_mode.addItem("下排对比（实验）：原图｜全局 H｜局部 H", "lower_device_plane")
        self.result_mode.currentIndexChanged.connect(self.refresh_result)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.valueChanged.connect(self.refresh_result)
        self.opacity_label = QLabel("待检图权重：50%")
        options_layout.addWidget(self.result_mode)
        options_layout.addWidget(self.opacity_label)
        options_layout.addWidget(self.opacity, 1)
        root.addWidget(options)

        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMaximumBlockCount(250)
        self.report_view.setPlainText("对齐报告会显示在这里。")
        report_box = QGroupBox("对齐报告（不是故障结论）")
        report_layout = QVBoxLayout(report_box)
        report_layout.addWidget(self.report_view)
        lower_splitter = QSplitter(Qt.Vertical)
        lower_splitter.addWidget(image_splitter)
        lower_splitter.addWidget(report_box)
        lower_splitter.setSizes([700, 170])
        root.addWidget(lower_splitter, 1)

    def _path_row(self, line_edit: QLineEdit, title: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("浏览…")
        button.clicked.connect(lambda: self.choose_image(line_edit, title))
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def choose_image(self, line_edit: QLineEdit, title: str) -> None:
        initial = line_edit.text() or str(CASE_DIR)
        selected, _ = QFileDialog.getOpenFileName(self, title, initial, IMAGE_FILTER)
        if selected:
            line_edit.setText(selected)

    def run_alignment(self) -> None:
        reference_path = Path(self.reference_path.text().strip())
        inspection_path = Path(self.inspection_path.text().strip())
        if not reference_path.is_file() or not inspection_path.is_file():
            QMessageBox.warning(self, "路径无效", "请分别选择存在的原图和待检图。")
            return
        try:
            self.setCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self._reference = read_image(reference_path)
            self._inspection = read_image(inspection_path)
            self.reference_panel.set_image(self._reference)
            self.inspection_panel.set_image(self._inspection)
            aligned, report = perspective.automatic_homography(self._reference, self._inspection)
            self._aligned = aligned
            self._report = report
            self._band_results = {}
            if aligned is None:
                self.report_view.setPlainText(format_alignment_report(report))
                self.result_panel.set_image(None)
                self.status.setText("对齐不可靠：第三栏不显示强行形变后的结果。")
                self.status.setStyleSheet("font-weight: 600; color: #b71c1c;")
            else:
                self._band_results = compute_local_band_results(self._reference, aligned)
                self.report_view.setPlainText(format_alignment_report(report) + format_local_band_report(self._band_results))
                self.refresh_result()
                self.status.setText("对齐完成：可在第三栏选择上/下排局部对齐对比；它不生成故障结论。")
                self.status.setStyleSheet("font-weight: 600; color: #1b5e20;")
        except Exception as error:  # Show a user-facing failure while preserving the source files.
            self._aligned = None
            self._band_results = {}
            self.result_panel.set_image(None)
            self.status.setText("读取或对齐失败。")
            self.status.setStyleSheet("font-weight: 600; color: #b71c1c;")
            QMessageBox.critical(self, "对齐失败", str(error))
        finally:
            self.unsetCursor()

    def refresh_result(self) -> None:
        if self._reference is None or self._aligned is None:
            return
        weight = self.opacity.value()
        self.opacity_label.setText(f"待检图权重：{weight}%")
        alpha = weight / 100.0
        mode = self.result_mode.currentData()
        if mode == "global_aligned":
            result = self._aligned
            self.result_panel.title.setText("③ 全局 H 对齐结果")
            self.result_panel.subtitle.setText("一张单一 3×3 单应矩阵将待检图映射回原图坐标。")
        elif mode == "global_blend":
            result = cv2.addWeighted(self._reference, 1.0 - alpha, self._aligned, alpha, 0)
            self.result_panel.title.setText("③ 全局 H 叠图")
            self.result_panel.subtitle.setText("原图与全局对齐后的待检图叠加；重影表示仍有视差、光照或真实变化。")
        elif mode == "band_preview":
            result = cv2.addWeighted(self._reference, 1.0 - alpha, self._aligned, alpha, 0)
            for band in self._band_results.values():
                x1, y1, x2, y2 = band["pixel_bounds"]
                chosen = band["local_crop"] if band["use_local_for_manual_preview"] else band["global_crop"]
                result[y1:y2, x1:x2] = cv2.addWeighted(band["reference_crop"], 1.0 - alpha, chosen, alpha, 0)
                colour = (30, 170, 30) if band["use_local_for_manual_preview"] else (0, 165, 255)
                cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), colour, 2)
            self.result_panel.title.setText("③ 分区预览（实验）")
            self.result_panel.subtitle.setText("绿框：该区局部 H 的边缘重合度更高；橙框：保留原全局 H。矩形外仍是全局 H。")
        elif mode in self._band_results:
            band = self._band_results[mode]
            name = "上排器件平面" if mode == "upper_device_plane" else "下排器件平面"
            local_state = "局部 H 被采用作预览" if band["use_local_for_manual_preview"] else "局部 H 未优于全局 H，预览仍保留供人工判断"
            result = np.hstack([
                labelled_panel(band["reference_crop"], "reference"),
                labelled_panel(cv2.addWeighted(band["reference_crop"], 1.0 - alpha, band["global_crop"], alpha, 0), "global H blend"),
                labelled_panel(cv2.addWeighted(band["reference_crop"], 1.0 - alpha, band["local_crop"], alpha, 0), "local H blend"),
            ])
            self.result_panel.title.setText(f"③ {name}：三栏局部对齐对比")
            self.result_panel.subtitle.setText(f"从左到右：原图、全局 H 叠图、局部 H 叠图。{local_state}。")
        else:
            return
        self.result_panel.set_image(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Optional reference image path.")
    parser.add_argument("--inspection", type=Path, help="Optional inspection image path.")
    parser.add_argument("--self-check", action="store_true", help="Run one read-only alignment and print JSON instead of opening the GUI.")
    parser.add_argument("--self-check-bands", action="store_true", help="Also calculate the read-only upper/lower local-band comparison report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = args.reference or DEFAULT_REFERENCE
    inspection_path = args.inspection or DEFAULT_INSPECTION
    if args.self_check:
        reference = read_image(reference_path)
        inspection = read_image(inspection_path)
        aligned, report = perspective.automatic_homography(reference, inspection)
        output: dict[str, Any] = {"aligned": aligned is not None, "report": report}
        if args.self_check_bands and aligned is not None:
            bands = compute_local_band_results(reference, aligned)
            output["bands"] = {
                name: {
                    key: value
                    for key, value in band.items()
                    if key not in {"reference_crop", "global_crop", "local_crop"}
                }
                for name, band in bands.items()
            }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = CabinetAlignmentViewer(reference_path, inspection_path)
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
