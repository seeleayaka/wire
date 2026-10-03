"""Stable fixed-camera assembly review using 2-3 templates and affine alignment."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

import assembly_template_review_app as base
from dimm_review_app_fixed import RoiCanvas


if not hasattr(RoiCanvas, "normalized_roi"):
    RoiCanvas.normalized_roi = RoiCanvas.roi  # type: ignore[attr-defined]


def multi_template_affine(
    reference: np.ndarray, inspection: np.ndarray, template_rois: list[list[float]]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.012, edgeThreshold=12)
    descriptors: list[np.ndarray] = []
    points: list[tuple[float, float]] = []
    owners: list[int] = []
    template_keypoints: list[int] = []
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    for index, roi in enumerate(template_rois, 1):
        x1, y1, x2, y2 = base.pixels(reference, roi)
        keypoints, desc = sift.detectAndCompute(gray_reference[y1:y2, x1:x2], None)
        template_keypoints.append(len(keypoints))
        if desc is None:
            continue
        descriptors.append(desc)
        points.extend((kp.pt[0] + x1, kp.pt[1] + y1) for kp in keypoints)
        owners.extend([index] * len(keypoints))
    report: dict[str, Any] = {
        "method": "multi_template_sift_affine",
        "template_count": len(template_rois),
        "template_keypoints": template_keypoints,
        "matches": 0,
        "inliers": 0,
    }
    if not descriptors:
        report["reason"] = "templates_have_too_little_texture"
        return None, report
    reference_desc = np.vstack(descriptors)
    inspection_kp, inspection_desc = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    report["inspection_keypoints"] = len(inspection_kp)
    if inspection_desc is None:
        report["reason"] = "inspection_has_too_little_texture"
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(inspection_desc, reference_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    report["matches"] = len(good)
    if len(good) < 24:
        report["reason"] = "too_few_template_matches"
        return None, report
    source = np.float32([inspection_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination = np.float32([points[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    affine, mask = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=4.0,
        maxIters=6000, confidence=0.995, refineIters=20,
    )
    if affine is None or mask is None:
        report["reason"] = "no_stable_affine_transform"
        return None, report
    inlier_mask = mask.ravel().astype(bool)
    report["inliers"] = int(inlier_mask.sum())
    report["inlier_ratio"] = round(report["inliers"] / len(good), 4)
    inlier_owners = [owners[match.trainIdx] for match, accepted in zip(good, inlier_mask) if accepted]
    report["inliers_by_template"] = {str(index): inlier_owners.count(index) for index in range(1, len(template_rois) + 1)}
    transformed = cv2.transform(source, affine)
    errors = np.linalg.norm(transformed.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
    report["median_reprojection_error"] = round(float(np.median(errors[inlier_mask])), 2) if report["inliers"] else None
    scale = float(np.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2))
    angle = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    report["scale"] = round(scale, 4)
    report["rotation_degrees"] = round(angle, 2)
    active_templates = sum(count >= 4 for count in report["inliers_by_template"].values())
    if (
        report["inliers"] < 20 or active_templates < 2
        or report["median_reprojection_error"] is None or report["median_reprojection_error"] > 3.0
        or not (0.80 <= scale <= 1.20) or abs(angle) > 15.0
    ):
        report["reason"] = "multi_template_alignment_unreliable_retake_or_reselect"
        return None, report
    height, width = reference.shape[:2]
    return cv2.warpAffine(inspection, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), report


def structural_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    overlay = aligned.copy()
    heat = np.zeros_like(reference)
    regions: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(rois, 1):
        x1, y1, x2, y2 = base.pixels(reference, roi)
        ref_gray = cv2.cvtColor(reference[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(aligned[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.GaussianBlur(cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX), (3, 3), 0)
        test_gray = cv2.GaussianBlur(cv2.normalize(test_gray, None, 0, 255, cv2.NORM_MINMAX), (3, 3), 0)
        tone = cv2.absdiff(ref_gray, test_gray)
        ref_edges = cv2.Canny(ref_gray, 45, 120)
        test_edges = cv2.Canny(test_gray, 45, 120)
        edge = cv2.absdiff(ref_edges, test_edges)
        score = np.maximum(tone, (edge * 0.80).astype(np.uint8))
        threshold = max(22.0, float(np.percentile(score, 98.0)))
        mask = np.uint8(score >= threshold) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        heat[y1:y2, x1:x2] = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
        minimum = max(180, int(mask.shape[0] * mask.shape[1] * 0.0010))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            regions.append({
                "check_region": roi_index, "left": x + x1, "top": y + y1,
                "right": x + x1 + width, "bottom": y + y1 + height,
                "area": round(area, 1), "structural_difference_score": round(float(score[y:y + height, x:x + width].mean()), 2),
            })
    regions.sort(key=lambda item: item["area"] * item["structural_difference_score"], reverse=True)
    for number, item in enumerate(regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 4)
        cv2.putText(overlay, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return overlay, heat, regions


class MultiTemplateReview(base.TemplateReview):
    def load_recipe(self) -> dict[str, Any]:
        recipe = super().load_recipe()
        if "template_rois" not in recipe:
            recipe["template_rois"] = [recipe["template_roi"]] if recipe.get("template_roi") else []
        return recipe

    def build(self) -> None:
        super().build()
        self.setWindowTitle("AI 装配视觉复核｜多模板稳定定位")
        left = self.centralWidget().layout().itemAt(0).widget()
        controls = left.layout()
        undo = QPushButton("撤回最后定位模板")
        undo.clicked.connect(self.undo_template)
        controls.insertWidget(3, undo)

    def refresh(self) -> None:
        count = len(self.recipe.get("template_rois", []))
        self.status.setText(f"定位模板：{count} 个（请设置 2–3 个）；检查区：{len(self.recipe.get('check_rois', []))} 个。")

    def choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择正确参考图", str(Path.home() / "Desktop"), "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.reference.setText(path)
            self.recipe = {"reference_image": path, "template_rois": [], "check_rois": []}
            self.save_recipe()
            self.refresh()

    def set_template(self) -> None:
        image = self.reference_image()
        if image is None:
            return
        picker = base.RectPicker(image, "添加稳定定位模板", "请在固定外壳、文字、螺丝孔或固定边缘框选。不要包含插头、线缆、插孔或会变化的部件。重复添加 2–3 个彼此分散的模板。", self)
        if picker.exec_() == picker.Accepted:
            templates = self.recipe.setdefault("template_rois", [])
            if len(templates) >= 3:
                QMessageBox.information(self, "模板数量已足够", "最多保存 3 个定位模板；如需重设，请先撤回或重置。")
                return
            templates.append(picker.roi())
            self.save_recipe()
            self.refresh()

    def undo_template(self) -> None:
        templates = self.recipe.get("template_rois", [])
        if templates:
            templates.pop()
            self.save_recipe()
            self.refresh()

    def reset_recipe(self) -> None:
        if QMessageBox.question(self, "确认重置", "删除当前定位模板和所有检查区域？", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.recipe["template_rois"] = []
            self.recipe["check_rois"] = []
            self.save_recipe()
            self.refresh()

    def run(self) -> None:
        ref_path = Path(self.reference.text().strip())
        test_path = Path(self.inspection.text().strip())
        templates = self.recipe.get("template_rois", [])
        if not ref_path.is_file() or not test_path.is_file() or len(templates) < 2 or not self.recipe.get("check_rois"):
            QMessageBox.warning(self, "设置未完成", "请选择两张图片，添加至少 2 个定位模板，并添加至少 1 个检查区域。")
            return
        reference, inspection = base.read_image(ref_path), base.read_image(test_path)
        aligned, alignment = multi_template_affine(reference, inspection, templates)
        output = base.OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
        output.mkdir(parents=True, exist_ok=True)
        if aligned is None:
            report = {"decision": "retake_or_reselect", "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
            self.status.setText("多模板定位不可靠：请重拍或重选 2–3 个分散、固定的模板。")
            return
        overlay, heat, regions = structural_regions(reference, aligned, self.recipe["check_rois"])
        base.write_image(output / "aligned.jpg", aligned)
        base.write_image(output / "anomaly_boxes.jpg", overlay)
        base.write_image(output / "check_heatmap.jpg", heat)
        report = {"decision": "manual_review_required", "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment, "template_rois": templates, "check_region_count": len(self.recipe["check_rois"]), "review_regions": regions, "note": "红框为检查区内的结构差异，需人工确认。"}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for view, name in zip(self.views, ("aligned.jpg", "anomaly_boxes.jpg", "check_heatmap.jpg")):
            view.load(output / name)
        self.tabs.setCurrentIndex(1)
        self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        self.status.setText(f"多模板定位成功；在 {len(self.recipe['check_rois'])} 个检查区发现 {len(regions)} 个结构差异。结果：{output}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MultiTemplateReview()
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
