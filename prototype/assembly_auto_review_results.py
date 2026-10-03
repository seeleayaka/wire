"""Automatic review UI with a clear three-state decision card."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

import assembly_auto_review_multiscale as multiscale

auto = multiscale.app


class ResultCardReview(auto.AutomaticReview):
    def build(self) -> None:
        super().build()
        left = self.centralWidget().layout().itemAt(0).widget()
        controls = left.layout()
        self.result_card = QFrame()
        self.result_card.setObjectName("resultCard")
        layout = QVBoxLayout(self.result_card)
        layout.setContentsMargins(13, 12, 13, 12)
        title = QLabel("判断结果")
        title.setStyleSheet("font-weight:bold;font-size:15px")
        self.result_headline = QLabel()
        self.result_headline.setStyleSheet("font-size:19px;font-weight:bold")
        self.result_detail = QLabel()
        self.result_detail.setWordWrap(True)
        buttons = QHBoxLayout()
        self.show_boxes_button = QPushButton("查看红框结果")
        self.show_boxes_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.open_folder_button = QPushButton("打开本次保存目录")
        self.open_folder_button.clicked.connect(self.open_result_folder)
        buttons.addWidget(self.show_boxes_button)
        buttons.addWidget(self.open_folder_button)
        layout.addWidget(title)
        layout.addWidget(self.result_headline)
        layout.addWidget(self.result_detail)
        layout.addLayout(buttons)
        controls.insertWidget(controls.count() - 1, self.result_card)
        self.current_output: Path | None = None
        self.set_decision("idle", "尚未检测", "请选择正确参考图和待检图后开始检测。")

    def set_decision(self, state: str, headline: str, detail: str) -> None:
        styles = {
            "idle": ("#f3f4f6", "#4b5563", "#d1d5db"),
            "processing": ("#eff6ff", "#2563eb", "#93c5fd"),
            "clear": ("#ecfdf5", "#15803d", "#86efac"),
            "review": ("#fff1f2", "#dc2626", "#fda4af"),
            "unreliable": ("#fffbeb", "#d97706", "#fcd34d"),
        }
        background, foreground, border = styles[state]
        self.result_card.setStyleSheet(f"QFrame#resultCard{{background:{background};border:1px solid {border};border-radius:8px}}")
        self.result_headline.setText(headline)
        self.result_headline.setStyleSheet(f"font-size:19px;font-weight:bold;color:{foreground}")
        self.result_detail.setText(detail)
        ready = state in {"clear", "review"}
        self.show_boxes_button.setEnabled(ready)
        self.open_folder_button.setEnabled(self.current_output is not None)

    def open_result_folder(self) -> None:
        if self.current_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_output)))

    def run(self) -> None:
        ref_path, test_path = Path(self.reference.text().strip()), Path(self.inspection.text().strip())
        if not ref_path.is_file() or not test_path.is_file() or not self.recipe.get("check_rois"):
            self.set_decision("idle", "尚未检测", "请选择两张图片，并至少添加一个检查区域。")
            return
        reference, inspection = auto.base.read_image(ref_path), auto.base.read_image(test_path)
        aligned, alignment = auto.automatic_affine(reference, inspection)
        output = auto.base.OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
        output.mkdir(parents=True, exist_ok=True)
        self.current_output = output
        if aligned is None:
            report = {"decision": "retake_photo", "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
            self.set_decision("unreliable", "无法可靠对齐，请重拍", f"定位未通过：{alignment.get('reason', '匹配不足')}。请固定拍摄高度、角度和范围后重拍。")
            self.status.setText("自动整机定位不可靠。")
            return
        overlay, heat, regions = auto.colour_aware_structural_regions(reference, aligned, self.recipe["check_rois"])
        auto.base.write_image(output / "aligned.jpg", aligned)
        auto.base.write_image(output / "anomaly_boxes.jpg", overlay)
        auto.base.write_image(output / "check_heatmap.jpg", heat)
        decision = "manual_review_required" if regions else "no_significant_difference"
        report = {"decision": decision, "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment, "check_region_count": len(self.recipe["check_rois"]), "review_regions": regions}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for view, name in zip(self.views, ("aligned.jpg", "anomaly_boxes.jpg", "check_heatmap.jpg")):
            view.load(output / name)
        self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        detail = f"定位通过 ｜ 检查区域：{len(self.recipe['check_rois'])} 个 ｜ 差异区域：{len(regions)} 个"
        if regions:
            self.tabs.setCurrentIndex(1)
            self.set_decision("review", "发现差异，请人工复核", detail)
            self.status.setText("发现差异，请人工复核。")
        else:
            self.tabs.setCurrentIndex(0)
            self.set_decision("clear", "未发现明显差异", detail)
            self.status.setText("未发现明显差异。")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ResultCardReview()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
