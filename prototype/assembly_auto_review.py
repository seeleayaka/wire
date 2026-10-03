"""Automatic full-frame registration with colour-aware inspection ROIs."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox, QPushButton

import assembly_template_review_app as base
from assembly_multi_template_review_color import colour_aware_structural_regions


def automatic_affine(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.014, edgeThreshold=12)
    ref_kp, ref_desc = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    test_kp, test_desc = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    report: dict[str, Any] = {"method": "automatic_full_frame_sift_affine", "reference_keypoints": len(ref_kp), "inspection_keypoints": len(test_kp), "matches": 0, "inliers": 0}
    if ref_desc is None or test_desc is None:
        report["reason"] = "too_little_visual_texture"
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(test_desc, ref_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    report["matches"] = len(good)
    if len(good) < 50:
        report["reason"] = "too_few_full_frame_matches"
        return None, report
    source = np.float32([test_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    affine, mask = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=10000, confidence=0.995, refineIters=30)
    if affine is None or mask is None:
        report["reason"] = "no_stable_full_frame_transform"
        return None, report
    inlier_mask = mask.ravel().astype(bool)
    report["inliers"] = int(inlier_mask.sum())
    report["inlier_ratio"] = round(report["inliers"] / len(good), 4)
    transformed = cv2.transform(source, affine)
    errors = np.linalg.norm(transformed.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
    report["median_reprojection_error"] = round(float(np.median(errors[inlier_mask])), 2) if report["inliers"] else None
    height, width = reference.shape[:2]
    cells = {(min(3, int(destination[i, 0, 0] * 4 / width)), min(3, int(destination[i, 0, 1] * 4 / height))) for i in np.where(inlier_mask)[0]}
    scale = float(np.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2))
    angle = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    report["inlier_grid_cells"] = len(cells)
    report["scale"] = round(scale, 4)
    report["rotation_degrees"] = round(angle, 2)
    if report["inliers"] < 45 or len(cells) < 3 or report["median_reprojection_error"] is None or report["median_reprojection_error"] > 3.0 or not (0.80 <= scale <= 1.20) or abs(angle) > 15.0:
        report["reason"] = "automatic_alignment_unreliable_retake_photo"
        return None, report
    return cv2.warpAffine(inspection, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), report


class AutomaticReview(base.TemplateReview):
    def load_recipe(self) -> dict[str, Any]:
        recipe = super().load_recipe()
        recipe.pop("template_roi", None)
        recipe.pop("template_rois", None)
        recipe.setdefault("check_rois", [])
        return recipe

    def build(self) -> None:
        super().build()
        self.setWindowTitle("AI 装配视觉复核｜自动整机定位")
        for button in self.findChildren(QPushButton):
            if "模板" in button.text():
                button.hide()
        for label in self.findChildren(QLabel):
            if "首次设置" in label.text():
                label.setText("首次设置：选择正确参考图，再添加真正要检查的插头、DIMM 或端子区域。日常待检时程序会自动利用整张图的固定特征定位，不需要框选模板。")

    def refresh(self) -> None:
        self.status.setText(f"自动整机定位已启用；检查区：{len(self.recipe.get('check_rois', []))} 个。")

    def choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择正确参考图", str(Path.home() / "Desktop"), "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.reference.setText(path)
            self.recipe = {"reference_image": path, "check_rois": []}
            self.save_recipe()
            self.refresh()

    def run(self) -> None:
        ref_path, test_path = Path(self.reference.text().strip()), Path(self.inspection.text().strip())
        if not ref_path.is_file() or not test_path.is_file() or not self.recipe.get("check_rois"):
            QMessageBox.warning(self, "设置未完成", "请选择正确参考图和待检图，并添加至少一个检查区域。")
            return
        reference, inspection = base.read_image(ref_path), base.read_image(test_path)
        aligned, alignment = automatic_affine(reference, inspection)
        output = base.OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
        output.mkdir(parents=True, exist_ok=True)
        if aligned is None:
            report = {"decision": "retake_photo", "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
            self.status.setText("自动整机定位不可靠：请固定拍摄位置后重拍。")
            return
        overlay, heat, regions = colour_aware_structural_regions(reference, aligned, self.recipe["check_rois"])
        base.write_image(output / "aligned.jpg", aligned)
        base.write_image(output / "anomaly_boxes.jpg", overlay)
        base.write_image(output / "check_heatmap.jpg", heat)
        report = {"decision": "manual_review_required", "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment, "check_region_count": len(self.recipe["check_rois"]), "review_regions": regions, "note": "自动整机定位后，仅在预设检查区内标出综合色彩和结构差异。"}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for view, name in zip(self.views, ("aligned.jpg", "anomaly_boxes.jpg", "check_heatmap.jpg")):
            view.load(output / name)
        self.tabs.setCurrentIndex(1)
        self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        self.status.setText(f"自动整机定位成功；在 {len(self.recipe['check_rois'])} 个检查区发现 {len(regions)} 个待确认差异。结果：{output}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AutomaticReview()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
