"""Two-image visual review with local alignment confidence and persistent-change scoring.

This version keeps every object in the check region.  It uses the whole image for
coarse registration, then makes a strictly limited local correction per check area.
Fine residual edges are reported as *uncertain* instead of being promoted to a red
alarm.  A real object-sized change remains a candidate after the local correction.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import assembly_auto_review_results as result_ui

auto = result_ui.auto
LAST_DIAGNOSTICS: list[dict[str, Any]] = []


def _illumination_normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    background = cv2.GaussianBlur(gray, (0, 0), 18.0)
    normalized = cv2.divide(gray, background + 0.03)
    return cv2.GaussianBlur(cv2.normalize(normalized, None, 0.0, 1.0, cv2.NORM_MINMAX), (5, 5), 0)


def _ssim_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    mu1, mu2 = cv2.GaussianBlur(first, (0, 0), 5.0), cv2.GaussianBlur(second, (0, 0), 5.0)
    sigma1 = cv2.GaussianBlur(first * first, (0, 0), 5.0) - mu1 * mu1
    sigma2 = cv2.GaussianBlur(second * second, (0, 0), 5.0) - mu2 * mu2
    sigma12 = cv2.GaussianBlur(first * second, (0, 0), 5.0) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2) + 1e-7)
    return np.clip((1.0 - ssim) * 127.5, 0.0, 255.0)


def _gradient(image: np.ndarray) -> np.ndarray:
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(x, y)


def _limited_local_ecc(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine only a small local residual; never permit an object-sized warp."""
    reference_gray, inspection_gray = _illumination_normalized_gray(reference), _illumination_normalized_gray(inspection)
    warp = np.eye(2, 3, dtype=np.float32)
    diagnostic: dict[str, Any] = {"method": "local_ecc_affine", "accepted": False, "correlation": None}
    try:
        correlation, warp = cv2.findTransformECC(
            reference_gray,
            inspection_gray,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-7),
            None,
            5,
        )
        scale_x = float(np.hypot(warp[0, 0], warp[1, 0]))
        scale_y = float(np.hypot(warp[0, 1], warp[1, 1]))
        angle = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
        translation = float(np.linalg.norm(warp[:, 2]))
        diagnostic.update(
            correlation=round(float(correlation), 4),
            translation_pixels=round(translation, 2),
            rotation_degrees=round(angle, 2),
            scale_x=round(scale_x, 4),
            scale_y=round(scale_y, 4),
        )
        # The limits deliberately prevent ECC from explaining away a moved DIMM.
        accepted = correlation >= 0.70 and translation <= 10.0 and abs(angle) <= 2.5 and 0.97 <= scale_x <= 1.03 and 0.97 <= scale_y <= 1.03
        diagnostic["accepted"] = bool(accepted)
        if accepted:
            local = cv2.warpAffine(inspection, warp, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
            valid = cv2.warpAffine(np.full(reference_gray.shape, 255, np.uint8), warp, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
            return local, valid, diagnostic
    except cv2.error as error:
        diagnostic["reason"] = "ecc_not_converged"
        diagnostic["opencv_error"] = str(error).split("\n", 1)[0]
    return inspection.copy(), np.full(reference_gray.shape, 255, np.uint8), diagnostic


def _candidate_regions(score: np.ndarray, valid: np.ndarray, roi_index: int, origin: tuple[int, int], diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    usable = score[valid > 0]
    if usable.size < 100:
        return []
    # A persistent object affects a neighbourhood.  Thin alignment halos have a
    # high edge score but weak 31-pixel neighbourhood support.
    neighbourhood = cv2.GaussianBlur(score, (31, 31), 0)
    detail_threshold = max(24.0, float(np.percentile(usable, 97.0)))
    broad_threshold = max(11.0, float(np.percentile(neighbourhood[valid > 0], 90.0)))
    detail = np.uint8((score >= detail_threshold) & (valid > 0)) * 255
    broad = np.uint8((neighbourhood >= broad_threshold) & (valid > 0)) * 255
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    detail = cv2.dilate(cv2.morphologyEx(detail, cv2.MORPH_OPEN, small), small, iterations=1)
    broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, large)
    mask = cv2.bitwise_or(detail, broad)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum = max(260, int(mask.shape[0] * mask.shape[1] * 0.0008))
    x0, y0 = origin
    regions: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        part_score = score[y:y + height, x:x + width]
        part_support = neighbourhood[y:y + height, x:x + width]
        mean_score = float(part_score.mean())
        mean_support = float(part_support.mean())
        # Very thin shapes are retained but are not high-confidence alone: this
        # keeps cable/clip changes visible without letting fin-edge halos dominate.
        thin = min(width, height) < 9 or (max(width, height) / max(1, min(width, height))) > 12.0
        high_confidence = bool(diagnostic.get("accepted") and diagnostic.get("correlation", 0.0) >= 0.78 and mean_support >= 18.0 and not thin)
        regions.append({
            "check_region": roi_index,
            "left": x + x0,
            "top": y + y0,
            "right": x + x0 + width,
            "bottom": y + y0 + height,
            "area": round(area, 1),
            "difference_score": round(mean_score, 2),
            "neighbourhood_score": round(mean_support, 2),
            "confidence": "high" if high_confidence else "uncertain",
            "shape": "thin_or_repetitive" if thin else "object_sized",
        })
    return regions


def robust_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    global LAST_DIAGNOSTICS
    LAST_DIAGNOSTICS = []
    overlay, heat, regions = aligned.copy(), np.zeros_like(reference), []
    for roi_index, roi in enumerate(rois, 1):
        x1, y1, x2, y2 = auto.base.pixels(reference, roi)
        reference_part, inspection_part = reference[y1:y2, x1:x2], aligned[y1:y2, x1:x2]
        local, valid, diagnostic = _limited_local_ecc(reference_part, inspection_part)
        diagnostic.update(check_region=roi_index, bounds=[x1, y1, x2, y2])
        LAST_DIAGNOSTICS.append(diagnostic)
        reference_gray, local_gray = _illumination_normalized_gray(reference_part), _illumination_normalized_gray(local)
        structure = _ssim_difference(reference_gray, local_gray)
        gradient = np.clip(np.abs(_gradient(reference_gray) - _gradient(local_gray)) * 255.0, 0.0, 255.0)
        reference_lab = cv2.cvtColor(reference_part, cv2.COLOR_BGR2LAB).astype(np.float32)
        local_lab = cv2.cvtColor(local, cv2.COLOR_BGR2LAB).astype(np.float32)
        colour = np.clip(np.linalg.norm(reference_lab - local_lab, axis=2) * 1.20, 0.0, 255.0)
        score = np.clip(structure * 0.55 + colour * 0.25 + gradient * 0.20, 0.0, 255.0).astype(np.float32)
        score[valid == 0] = 0.0
        heat[y1:y2, x1:x2] = cv2.applyColorMap(np.uint8(score), cv2.COLORMAP_JET)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
        regions.extend(_candidate_regions(score, valid, roi_index, (x1, y1), diagnostic))
    regions.sort(key=lambda item: (item["confidence"] == "high", item["area"] * item["neighbourhood_score"]), reverse=True)
    for number, item in enumerate(regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        colour = (0, 0, 255) if item["confidence"] == "high" else (0, 180, 255)
        cv2.rectangle(overlay, (left, top), (right, bottom), colour, 4)
        cv2.putText(overlay, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 3)
    return overlay, heat, regions


auto.colour_aware_structural_regions = robust_regions


class RobustReview(result_ui.ResultCardReview):
    def set_decision(self, state: str, headline: str, detail: str) -> None:
        if state == "uncertain":
            super().set_decision("unreliable", headline, detail)
            self.show_boxes_button.setEnabled(True)
            return
        super().set_decision(state, headline, detail)

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
            report = {"decision": "alignment_uncertain_manual_review", "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
            self.set_decision("uncertain", "定位不够可靠，请人工复核", f"全图定位未通过：{alignment.get('reason', '匹配不足')}。重拍可改善，但并非必须。")
            self.status.setText("全图定位不够可靠：结果已保存，建议人工复核。")
            return
        overlay, heat, regions = robust_regions(reference, aligned, self.recipe["check_rois"])
        auto.base.write_image(output / "aligned.jpg", aligned)
        auto.base.write_image(output / "anomaly_boxes.jpg", overlay)
        auto.base.write_image(output / "check_heatmap.jpg", heat)
        high = sum(item["confidence"] == "high" for item in regions)
        uncertain = len(regions) - high
        report = {
            "decision": "confirmed_difference" if high else ("uncertain_difference_manual_review" if regions else "no_significant_difference"),
            "reference": str(ref_path), "inspection": str(test_path), "alignment": alignment,
            "local_alignment": LAST_DIAGNOSTICS, "check_region_count": len(self.recipe["check_rois"]),
            "high_confidence_regions": high, "uncertain_regions": uncertain, "review_regions": regions,
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for view, name in zip(self.views, ("aligned.jpg", "anomaly_boxes.jpg", "check_heatmap.jpg")):
            view.load(output / name)
        self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        detail = f"全图定位通过 ｜ 检查区域：{len(self.recipe['check_rois'])} 个 ｜ 高可信：{high} 个 ｜ 待复核：{uncertain} 个"
        if high:
            self.tabs.setCurrentIndex(1)
            self.set_decision("review", "发现可信差异，请人工复核", detail)
            self.status.setText("发现可信差异：红框为高可信，橙框为待复核。")
        elif regions:
            self.tabs.setCurrentIndex(1)
            self.set_decision("uncertain", "发现待复核差异", detail)
            self.status.setText("局部对齐或差异可信度不足：请人工复核橙框区域。")
        else:
            self.tabs.setCurrentIndex(0)
            self.set_decision("clear", "未发现明显差异", detail)
            self.status.setText("未发现明显差异。")


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = RobustReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
