"""Interactive local preview for the current wire-harness segmentation weight.

This remains an evidence viewer. It never updates an assembly template, runs
topology, or returns an inspection verdict.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# On this Windows host the model runtime must load before PyQt creates a DLL
# context. Import through Ultralytics, matching the already verified POC path.
from ultralytics import YOLO  # noqa: F401

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from wire_harness_segmentation_poc import (
    DEFAULT_LOCAL_MODEL,
    ROOT,
    build_report,
    infer_local,
    read_image,
    save_artifacts,
)


OUTPUT_ROOT = ROOT / "output" / "model_preview"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)"


def summary_text(report: dict) -> str:
    counts = report["summary"]["target_class_counts"]
    return (
        f"Predictions: {report['summary']['prediction_count']}\n"
        f"Cable: {counts['cable']}    Connector: {counts['connector']}\n"
        f"Clip: {counts['clip']}    Strap: {counts['strap']}"
    )


class ImagePanel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(460, 350)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background: #171b20; color: #c8d0d8; padding: 8px;")

    def set_image(self, path: str | Path) -> None:
        image = QImage(str(path))
        if image.isNull():
            raise ValueError(f"Cannot display image: {path}")
        self._image = image
        self._render()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)


class ModelPreview(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.current_artifacts: dict[str, str] = {}
        self.setWindowTitle("Wire Harness Model Preview")
        self.resize(1420, 920)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        controls = QWidget()
        form = QFormLayout(controls)
        form.setContentsMargins(0, 0, 0, 0)

        self.model_path = QLineEdit(str(DEFAULT_LOCAL_MODEL))
        model_button = QPushButton("Choose model")
        model_button.clicked.connect(self.choose_model)
        form.addRow("Model", self._path_row(self.model_path, model_button))

        self.image_path = QLineEdit()
        image_button = QPushButton("Choose image")
        image_button.clicked.connect(self.choose_image)
        form.addRow("Image", self._path_row(self.image_path, image_button))

        options = QWidget()
        options_layout = QHBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.25)
        self.confidence.setDecimals(2)
        self.image_size = QSpinBox()
        self.image_size.setRange(320, 1536)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(960)
        self.mask_choice = QComboBox()
        self.mask_choice.addItems(["Cable mask", "All target mask", "Connector mask", "Clip mask", "Strap mask"])
        self.mask_choice.currentIndexChanged.connect(self.show_selected_mask)
        self.run_button = QPushButton("Run preview")
        self.run_button.clicked.connect(self.run_preview)
        options_layout.addWidget(QLabel("Confidence"))
        options_layout.addWidget(self.confidence)
        options_layout.addSpacing(12)
        options_layout.addWidget(QLabel("Image size"))
        options_layout.addWidget(self.image_size)
        options_layout.addSpacing(12)
        options_layout.addWidget(QLabel("Mask"))
        options_layout.addWidget(self.mask_choice)
        options_layout.addStretch(1)
        options_layout.addWidget(self.run_button)
        form.addRow("Options", options)
        layout.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.source_view = ImagePanel("Select an image to preview")
        self.overlay_view = ImagePanel("Segmentation overlay")
        self.mask_view = ImagePanel("Selected mask")
        splitter.addWidget(self.source_view)
        splitter.addWidget(self.overlay_view)
        splitter.addWidget(self.mask_view)
        splitter.setSizes([470, 470, 470])
        layout.addWidget(splitter, 1)

        lower = QSplitter(Qt.Horizontal)
        self.summary = QLabel("No local inference has run.")
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary.setMinimumWidth(300)
        self.summary.setStyleSheet("background: #f2f4f6; border: 1px solid #c9d0d7; padding: 10px;")
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(170)
        lower.addWidget(self.summary)
        lower.addWidget(self.report)
        lower.setSizes([360, 1040])
        layout.addWidget(lower)

        self.statusBar().showMessage("Local evidence preview only")

    @staticmethod
    def _path_row(line_edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def choose_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Choose YOLO segmentation model", self.model_path.text(), "PyTorch weights (*.pt)")
        if selected:
            self.model_path.setText(selected)

    def choose_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Choose image", self.image_path.text(), IMAGE_FILTER)
        if selected:
            self.image_path.setText(selected)
            self.source_view.set_image(selected)

    def run_preview(self) -> None:
        image_path = Path(self.image_path.text().strip())
        model_path = Path(self.model_path.text().strip())
        if not image_path.is_file():
            QMessageBox.warning(self, "Image missing", "Choose a readable image first.")
            return
        if not model_path.is_file():
            QMessageBox.warning(self, "Model missing", "Choose a compatible local .pt weight first.")
            return

        self.run_button.setEnabled(False)
        self.statusBar().showMessage("Running local segmentation...")
        QApplication.processEvents()
        try:
            image = read_image(image_path)
            records, warnings = infer_local(
                image_path,
                model_path,
                confidence_threshold=float(self.confidence.value()),
                image_size=int(self.image_size.value()),
            )
            report = build_report(
                image_path,
                image,
                records,
                warnings,
                confidence_threshold=float(self.confidence.value()),
                model_id=str(model_path),
                source="ultralytics_local_preview",
            )
            output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_artifacts = save_artifacts(output_dir, image, report)
            self.source_view.set_image(image_path)
            self.overlay_view.set_image(self.current_artifacts["overlay"])
            self.show_selected_mask()
            self.summary.setText(summary_text(report) + f"\n\nOutput:\n{output_dir}")
            self.report.setPlainText(json.dumps({
                "model": str(model_path),
                "input": str(image_path),
                "summary": report["summary"],
                "predictions": [
                    {"class": item["source_class"], "confidence": item["confidence"], "box_xyxy": item["box_xyxy"]}
                    for item in records
                ],
                "warnings": report["warnings"],
            }, ensure_ascii=False, indent=2))
            self.statusBar().showMessage(f"Saved preview artifacts to {output_dir}")
        except Exception as error:
            QMessageBox.critical(self, "Preview failed", str(error))
            self.statusBar().showMessage("Preview failed")
        finally:
            self.run_button.setEnabled(True)

    def show_selected_mask(self) -> None:
        if not self.current_artifacts:
            return
        mask_key = {
            "Cable mask": "mask_cable",
            "All target mask": "mask_all_target",
            "Connector mask": "mask_connector",
            "Clip mask": "mask_clip",
            "Strap mask": "mask_strap",
        }[self.mask_choice.currentText()]
        self.mask_view.set_image(self.current_artifacts[mask_key])


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ModelPreview()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
