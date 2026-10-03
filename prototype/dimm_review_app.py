"""Chinese desktop UI for fixed-view, human-reviewed right-upper DIMM checks."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from dimm_upper_right_review import align_to, normalize_gray, read_image, regions_from_difference, roi_pixels, write_image


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "dimm_review_app.json"
OUTPUT_ROOT = ROOT / "output" / "dimm_reviews"
NAMES = ("主参考（right3）", "备用参考（right4）")


def config_default() -> dict[str, Any]:
    return {"scope": "right_upper_dimm_bank_only", "output_root": str(OUTPUT_ROOT), "profiles": [{"name": n, "reference_image": "", "dimm_bank_roi": None} for n in NAMES]}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return config_default()
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data.setdefault("output_root", str(OUTPUT_ROOT))
    profiles = data.setdefault("profiles", [])
    while len(profiles) < 2:
        profiles.append({"name": NAMES[len(profiles)], "reference_image": "", "dimm_bank_roi": None})
    return data


def save_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def label_roi(path: Path) -> list[float] | None:
    image = read_image(path)
    boxes = cv2.selectROIs("框选右上 DIMM 槽组：Enter/空格确认，Esc取消", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if len(boxes) != 1:
        return None
    x, y, width, height = (int(v) for v in boxes[0])
    h, w = image.shape[:2]
    return [round(x / w, 6), round(y / h, 6), round((x + width) / w, 6), round((y + height) / h, 6)]


def full_overview(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(normalize_gray(reference), normalize_gray(aligned))
    mask = np.uint8(diff >= max(35.0, float(np.percentile(diff, 99.7)))) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), kernel, iterations=2)
    result = aligned.copy()
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    result[mask > 0] = cv2.addWeighted(aligned, 0.68, heatmap, 0.32, 0)[mask > 0]
    return result


def inspect(profile: dict[str, Any], image_path: Path, output: Path) -> dict[str, Any] | None:
    reference = read_image(Path(profile["reference_image"]))
    aligned, alignment = align_to(reference, read_image(image_path))
    if aligned is None:
        return None
    difference, regions, threshold = regions_from_difference(reference, aligned, profile["dimm_bank_roi"])
    x1, y1, x2, y2 = roi_pixels(reference, profile["dimm_bank_roi"])
    overlay = aligned.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 4)
    for number, region in enumerate(regions, 1):
        left, top, right, bottom = (int(region[k]) for k in ("left", "top", "right", "bottom"))
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 5)
        cv2.putText(overlay, str(number), (left, max(40, top - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    panel = np.hstack([reference[y1:y2, x1:x2], aligned[y1:y2, x1:x2], overlay[y1:y2, x1:x2], cv2.applyColorMap(difference, cv2.COLORMAP_JET)])
    write_image(output / "aligned.jpg", aligned)
    write_image(output / "full_board_overview.jpg", full_overview(reference, aligned))
    write_image(output / "right_upper_overview.jpg", overlay)
    write_image(output / "right_upper_review_panel.jpg", panel)
    return {"profile": profile["name"], "reference": profile["reference_image"], "alignment": alignment, "difference_threshold": round(threshold, 2), "review_region_count": len(regions), "review_regions": regions}


class ImageView(QScrollArea):
    def __init__(self, message: str) -> None:
        super().__init__(); self.setWidgetResizable(True)
        self.label = QLabel(message); self.label.setAlignment(Qt.AlignCenter); self.label.setStyleSheet("color:#666; padding:18px")
        self.setWidget(self.label)
    def load(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        self.label.setPixmap(pixmap) if not pixmap.isNull() else self.label.setText(f"无法载入：{path}")
        if not pixmap.isNull(): self.label.resize(pixmap.size())


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.data = load_config(); self.setWindowTitle("AI 装配视觉复核｜右上 DIMM 第一版"); self.resize(1480, 920); self.build(); self.load_form()

    def build(self) -> None:
        root = QWidget(); self.setCentralWidget(root); layout = QHBoxLayout(root); split = QSplitter(Qt.Horizontal); layout.addWidget(split)
        left = QWidget(); left.setMinimumWidth(360); left.setMaximumWidth(480); form_layout = QVBoxLayout(left)
        intro = QLabel("第一版：只检测右上 DIMM 槽组。\n全盘总览仅提示变化；所有结果必须人工确认。")
        intro.setWordWrap(True); intro.setStyleSheet("background:#eef6ff; padding:12px; border-radius:6px"); form_layout.addWidget(intro)
        group = QGroupBox("正确参考图（不同角度分别标记）"); grid = QFormLayout(group); self.refs: list[QLineEdit] = []
        for index, title in enumerate(("主参考图 right3", "备用参考图 right4")):
            row = QWidget(); box = QHBoxLayout(row); box.setContentsMargins(0, 0, 0, 0); edit = QLineEdit(); choose = QPushButton("选择"); choose.clicked.connect(lambda _=False, i=index: self.choose_ref(i)); box.addWidget(edit); box.addWidget(choose); grid.addRow(title, row); self.refs.append(edit)
            mark = QPushButton(f"框选{title}的右上 DIMM 槽组"); mark.clicked.connect(lambda _=False, i=index: self.mark(i)); grid.addRow("", mark)
        form_layout.addWidget(group)
        photo_group = QGroupBox("待检图片"); photo_form = QFormLayout(photo_group); row = QWidget(); box = QHBoxLayout(row); box.setContentsMargins(0,0,0,0); self.photo = QLineEdit(); choose_photo = QPushButton("选择"); choose_photo.clicked.connect(self.choose_photo); box.addWidget(self.photo); box.addWidget(choose_photo); photo_form.addRow("待检图片", row); form_layout.addWidget(photo_group)
        run = QPushButton("开始检测并保存人工复核报告"); run.setMinimumHeight(48); run.setStyleSheet("font-size:16px; font-weight:bold; background:#1677ff; color:white"); run.clicked.connect(self.run); form_layout.addWidget(run)
        self.status = QLabel("先选择 right3、right4，并分别框选右上 DIMM 槽组。"); self.status.setWordWrap(True); form_layout.addWidget(self.status); form_layout.addStretch(); split.addWidget(left)
        right = QWidget(); right_layout = QVBoxLayout(right); self.tabs = QTabWidget(); self.views = [ImageView("尚未检测") for _ in range(4)]
        for view, title in zip(self.views, ("全盘对齐", "全盘总览", "右上 DIMM 红框", "右上 DIMM 对比")): self.tabs.addTab(view, title)
        right_layout.addWidget(self.tabs, 3); self.report = QTextEdit(); self.report.setReadOnly(True); self.report.setPlaceholderText("检测结果和报告会显示在这里。"); right_layout.addWidget(self.report, 1); split.addWidget(right); split.setSizes([420, 1060])

    def load_form(self) -> None:
        for index, edit in enumerate(self.refs): edit.setText(self.data["profiles"][index].get("reference_image", ""))

    def sync(self) -> None:
        for index, edit in enumerate(self.refs):
            value = edit.text().strip(); profile = self.data["profiles"][index]
            if value != profile.get("reference_image", ""): profile["reference_image"], profile["dimm_bank_roi"] = value, None
        save_config(self.data)

    def choose_ref(self, index: int) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择正确参考图", str(Path.home() / "Desktop"), "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path: self.refs[index].setText(path); self.sync(); self.status.setText("参考图已更新，请重新框选该角度的右上 DIMM 槽组。")

    def mark(self, index: int) -> None:
        self.sync(); profile = self.data["profiles"][index]; path = Path(profile.get("reference_image", ""))
        if not path.is_file(): QMessageBox.warning(self, "缺少参考图", "请先选择有效的正确参考图。"); return
        try: roi = label_roi(path)
        except Exception as error: QMessageBox.critical(self, "无法框选", str(error)); return
        if roi is None: self.status.setText("未保存区域。请重新框选一个完整的槽组。"); return
        profile["dimm_bank_roi"] = roi; save_config(self.data); self.status.setText(f"已保存 {profile['name']} 的 DIMM 槽组。")

    def choose_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择待检图片", str(Path.home() / "Desktop"), "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path: self.photo.setText(path)

    def run(self) -> None:
        self.sync(); image = Path(self.photo.text().strip()); profiles = [p for p in self.data["profiles"] if p.get("reference_image") and p.get("dimm_bank_roi")]
        if not image.is_file(): QMessageBox.warning(self, "缺少待检图片", "请选择有效的待检图片。"); return
        if not profiles: QMessageBox.warning(self, "未完成初始化", "请至少标记一张正确参考图的右上 DIMM 槽组。"); return
        output = Path(self.data["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S"); output.mkdir(parents=True, exist_ok=True); candidates, attempts = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor); QApplication.processEvents()
        try:
            for number, profile in enumerate(profiles):
                folder = output / ("primary" if number == 0 else "backup")
                try:
                    result = inspect(profile, image, folder)
                    if result is None: attempts.append({"profile":profile["name"],"result":"alignment_failed"}); continue
                    a = result["alignment"]; candidates.append((float(a["inliers"])*float(a["inlier_ratio"]), result, folder))
                except Exception as error: attempts.append({"profile":profile["name"],"result":"error","detail":str(error)})
            if not candidates:
                report = {"decision":"retake_photo","inspection":str(image),"attempts":attempts,"reason":"两张参考图都无法稳定对齐，请按固定位置、方向和距离重拍。"}; (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); self.report.setPlainText(json.dumps(report,ensure_ascii=False,indent=2)); self.status.setText("无法稳定对齐，请重拍。"); return
            _, result, folder = max(candidates, key=lambda item:item[0]); report = {"prototype":True,"decision":"manual_review_required","scope":"right_upper_dimm_bank_only","inspection":str(image),"selected_reference":result["profile"],"attempts":attempts,**result,"note":"红框只表示视觉变化，必须人工确认内存条是否移位、未插紧或插错。"}; (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            for view, name in zip(self.views, ("aligned.jpg","full_board_overview.jpg","right_upper_overview.jpg","right_upper_review_panel.jpg")): view.load(folder/name)
            self.report.setPlainText(json.dumps(report,ensure_ascii=False,indent=2)); self.status.setText(f"检测完成：使用{result['profile']}；右上 DIMM 有 {result['review_region_count']} 个复核区。报告：{output}")
        finally: QApplication.restoreOverrideCursor()


def main() -> None:
    app = QApplication(sys.argv); app.setStyle("Fusion"); window = Window(); window.show(); raise SystemExit(app.exec_())


if __name__ == "__main__": main()
