"""DINO-fused, non-semantic visual review.

The operator sees only yellow possible-difference rectangles.  Conventional image
differences and DINOv2 patch differences stay in the report for later regression
analysis; neither is presented as a component-name or fault diagnosis.
"""
from __future__ import annotations

import json
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# DINO imports Torch.  On Windows Torch must be imported before Qt creates its
# application/DLL context, otherwise loading c10.dll can fail with WinError 1114.
from tiled_dino_review import review_components

import cv2
import numpy as np
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from assembly_anchor_review_app import ZoomImageView
import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v6 as adaptive
import deepseek_mask_review
import sam3_wire_fusion
from capture_advice import build_capture_advice, render_capture_advice


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection_agent import InspectionTask, WorkflowError  # noqa: E402
from inspection_agent import gui_bridge as inspection_agent_bridge  # noqa: E402
from inspection_agent.local_review_gui_bridge import import_for_current_view, task_fingerprint  # noqa: E402
from inspection_agent.optional_port_crop_review import sha as port_input_sha, SCENE as PORT_SCENE  # noqa: E402
from inspection_agent.port_crop_gui_bridge import run_gui_port_review, render_port_overlay  # noqa: E402

CABINET_RECIPE = ROOT / "config" / "cabinet_dino_review_recipe.json"


def dino_fused_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run DINO fusion in every selected review area after global alignment."""
    merged = adaptive._merge_rois(rois)
    # Keep the existing local-alignment diagnostics as a gate/report, but use a
    # semantic-free DINO+traditional score for the operator-facing candidates.
    _, fallback_heat, fallback_regions = adaptive.adaptive_regions(reference, aligned, merged)
    diagnostics = list(robust.LAST_DIAGNOSTICS)
    local_aligned = getattr(robust, "LAST_LOCAL_ALIGNED", None)
    local_accepted_mask = getattr(robust, "LAST_LOCAL_ACCEPTED_MASK", None)
    local_alignment_parts = getattr(robust, "LAST_LOCAL_ALIGNMENT_PARTS", None)
    overlay, heat, regions = aligned.copy(), np.zeros_like(reference), []
    try:
        for check_index, roi in enumerate(merged, 1):
            left, top, right, bottom = adaptive.local_review.motion.perspective.auto.base.pixels(reference, roi)
            roi_area_ratio = ((right - left) * (bottom - top)) / float(reference.shape[0] * reference.shape[1])
            # Large cabinet-wide regions contain unavoidable labels, rails and
            # residual edges. Smaller targeted checks retain tiny-defect sensitivity.
            require_cross_evidence = roi_area_ratio >= 0.50
            valid_mask = getattr(adaptive.robust.auto, "LAST_WARP_VALID_MASK", None)
            valid_part = valid_mask[top:bottom, left:right] if valid_mask is not None else None
            reference_part = reference[top:bottom, left:right]
            aligned_part = aligned[top:bottom, left:right]
            local_coverage = 0.0
            local_alignment_accepted = False
            if check_index <= len(diagnostics):
                local_coverage = float(diagnostics[check_index - 1].get("accepted_coverage", 0.0) or 0.0)
                local_alignment_accepted = bool(diagnostics[check_index - 1].get("accepted", False))
            local_part = None
            accepted_part = None
            if local_alignment_accepted and isinstance(local_alignment_parts, list) and check_index <= len(local_alignment_parts):
                stored_part = local_alignment_parts[check_index - 1]
                stored_aligned = stored_part.get("aligned")
                stored_mask = stored_part.get("accepted_mask")
                if (
                    stored_part.get("bounds") == [left, top, right, bottom]
                    and isinstance(stored_aligned, np.ndarray)
                    and isinstance(stored_mask, np.ndarray)
                    and stored_aligned.shape == aligned_part.shape
                    and stored_mask.shape == aligned_part.shape[:2]
                ):
                    local_part = stored_aligned
                    accepted_part = stored_mask > 0
            if (
                local_alignment_accepted
                and local_part is None
                and isinstance(local_aligned, np.ndarray)
                and isinstance(local_accepted_mask, np.ndarray)
                and local_aligned.shape[:2] == aligned.shape[:2]
                and local_accepted_mask.shape == aligned.shape[:2]
            ):
                local_part = local_aligned[top:bottom, left:right]
                accepted_part = local_accepted_mask[top:bottom, left:right] > 0
            use_local_alignment = local_part is not None and accepted_part is not None
            if use_local_alignment:
                aligned_part = aligned_part.copy()
                aligned_part[accepted_part] = local_part[accepted_part]
            # A large, operator-drawn check region remains one UI region.  The
            # internal tiled branch adds local-scale evidence so a small defect
            # is not diluted by the surrounding texture during ROI normalization.
            score, metadata, local_regions = review_components(
                reference_part,
                aligned_part,
                valid_part,
                reference_size=(reference.shape[1], reference.shape[0]),
                roi_area_ratio=roi_area_ratio,
                require_whole_cross_evidence=require_cross_evidence,
            )
            heat[top:bottom, left:right] = cv2.applyColorMap(np.uint8(np.clip(score, 0, 255)), cv2.COLORMAP_JET)
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 215, 255), 2)
            if check_index <= len(diagnostics):
                diagnostics[check_index - 1]["dino"] = metadata
                diagnostics[check_index - 1]["comparison_mode"] = "dino_traditional_fusion"
                diagnostics[check_index - 1]["roi_area_ratio"] = round(roi_area_ratio, 4)
                diagnostics[check_index - 1]["dino_alignment_input"] = "local_ecc_corrected" if use_local_alignment else "global_homography"
                diagnostics[check_index - 1]["dino_local_correction_coverage"] = round(local_coverage, 3)
                diagnostics[check_index - 1]["dino_local_alignment_accepted"] = local_alignment_accepted
            for item in local_regions:
                regions.append({
                    "check_region": check_index,
                    "left": int(item["left"]) + left,
                    "top": int(item["top"]) + top,
                    "right": int(item["right"]) + left,
                    "bottom": int(item["bottom"]) + top,
                    "area": item["area"],
                    "difference_score": item["difference_score"],
                    "confidence": "possible_difference",
                    "comparison_mode": "dino_traditional_fusion",
                    "source_tiles": item.get("source_tiles", ["whole_roi"]),
                    "evidence_summary": item.get("evidence_summary", {"whole_roi_overlap": True}),
                    "evidence_scores": item.get("evidence_scores", {}),
                })
    except Exception as error:
        # An unavailable/corrupt model must never leave the operator without a
        # review result.  Reuse the pre-DINO candidate branch and record why.
        for item in diagnostics:
            item["dino_error"] = str(error).split("\n", 1)[0]
            item["comparison_mode"] = "traditional_fallback_dino_unavailable"
            item["dino_available"] = False
        robust.LAST_DIAGNOSTICS = diagnostics
        return aligned.copy(), fallback_heat, fallback_regions

    robust.LAST_DIAGNOSTICS = diagnostics
    regions.sort(key=lambda item: item["area"] * item["difference_score"], reverse=True)
    for number, item in enumerate(regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 180, 255), 4)
        cv2.putText(overlay, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 3)
    return overlay, heat, regions


robust.robust_regions = dino_fused_regions


def _candidate_bbox(candidate: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    values = candidate.get("bbox_xyxy")
    if values is None:
        values = [candidate["left"], candidate["top"], candidate["right"], candidate["bottom"]]
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("candidate has no valid bbox_xyxy or left/top/right/bottom bounds")
    left, top, right, bottom = (int(value) for value in values)
    left, right = max(0, min(width, left)), max(0, min(width, right))
    top, bottom = max(0, min(height, top)), max(0, min(height, bottom))
    if right <= left or bottom <= top:
        raise ValueError("candidate bounds are outside the aligned image")
    return left, top, right, bottom


def _candidate_crop_bounds(
    candidate: dict[str, Any], width: int, height: int, *, context_ratio: float = 0.35
) -> tuple[int, int, int, int, tuple[int, int, int, int]]:
    left, top, right, bottom = _candidate_bbox(candidate, width, height)
    padding = max(24, round(max(right - left, bottom - top) * context_ratio))
    crop_left, crop_top = max(0, left - padding), max(0, top - padding)
    crop_right, crop_bottom = min(width, right + padding), min(height, bottom + padding)
    return crop_left, crop_top, crop_right, crop_bottom, (left, top, right, bottom)


def _candidate_comparison_records(
    reference_path: Path,
    aligned_path: Path,
    output_dir: Path,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Persist enlarged matched crops for manual reference-versus-inspection review."""
    reference = adaptive.robust.auto.base.read_image(reference_path)
    aligned = adaptive.robust.auto.base.read_image(aligned_path)
    if reference.shape != aligned.shape:
        raise ValueError("reference and aligned inspection images must have identical dimensions for candidate comparison")
    height, width = reference.shape[:2]
    comparison_dir = output_dir / "candidate_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        crop_left, crop_top, crop_right, crop_bottom, (left, top, right, bottom) = _candidate_crop_bounds(candidate, width, height)
        reference_crop = reference[crop_top:crop_bottom, crop_left:crop_right].copy()
        inspection_crop = aligned[crop_top:crop_bottom, crop_left:crop_right].copy()
        focus_left, focus_top = left - crop_left, top - crop_top
        focus_right, focus_bottom = right - crop_left, bottom - crop_top
        for image in (reference_crop, inspection_crop):
            cv2.rectangle(image, (focus_left, focus_top), (focus_right, focus_bottom), (0, 215, 255), 2)
        key = str(candidate.get("comparison_key", f"candidate_{index:02d}"))
        reference_crop_path = comparison_dir / f"{key}_reference.jpg"
        inspection_crop_path = comparison_dir / f"{key}_inspection_aligned.jpg"
        adaptive.robust.auto.base.write_image(reference_crop_path, reference_crop)
        adaptive.robust.auto.base.write_image(inspection_crop_path, inspection_crop)
        records.append({
            "label": str(candidate.get("comparison_label", f"候选 {index:02d}")),
            "reference_crop": reference_crop_path,
            "inspection_crop": inspection_crop_path,
        })
    return records


class CandidateComparisonPanel(QWidget):
    """Choose one candidate and inspect its reference/aligned-image crop pair."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[dict[str, Any]] = []
        layout = QVBoxLayout(self)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("候选区域", self))
        self.selector = QComboBox(self)
        self.selector.currentIndexChanged.connect(self._show_selection)
        selector_row.addWidget(self.selector, 1)
        layout.addLayout(selector_row)
        views = QHBoxLayout()
        reference_column = QWidget(self)
        reference_layout = QVBoxLayout(reference_column)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.addWidget(QLabel("参考图局部", reference_column))
        self.reference_view = ZoomImageView("等待候选区域")
        reference_layout.addWidget(self.reference_view, 1)
        inspection_column = QWidget(self)
        inspection_layout = QVBoxLayout(inspection_column)
        inspection_layout.setContentsMargins(0, 0, 0, 0)
        inspection_layout.addWidget(QLabel("待检图局部（已对齐）", inspection_column))
        self.inspection_view = ZoomImageView("等待候选区域")
        inspection_layout.addWidget(self.inspection_view, 1)
        views.addWidget(reference_column, 1)
        views.addWidget(inspection_column, 1)
        layout.addLayout(views, 1)

    def set_records(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems([str(record["label"]) for record in records])
        self.selector.blockSignals(False)
        if records:
            self._show_selection(0)
            return
        for view in (self.reference_view, self.inspection_view):
            view.scene.clear()
            view.item = None
            view.scene.addText("等待候选区域")

    def _show_selection(self, index: int) -> None:
        if not 0 <= index < len(self._records):
            return
        record = self._records[index]
        self.reference_view.load(Path(record["reference_crop"]))
        self.inspection_view.load(Path(record["inspection_crop"]))


class BusyIndicator(QWidget):
    """Small indeterminate spinner used only while a real worker is active."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setFixedSize(18, 18)
        self.hide()

    def set_running(self, running: bool) -> None:
        if running:
            self.show()
            if not self._timer.isActive():
                self._timer.start(70)
        else:
            self._timer.stop()
            self.hide()

    def _advance(self) -> None:
        self._angle = (self._angle + 28) % 360
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#1677ff"), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(self.rect().adjusted(3, 3, -3, -3), self._angle * 16, 105 * 16)


class InitialReviewWorker(QThread):
    """Run image loading, registration and DINO extraction off the Qt thread."""

    stage_changed = pyqtSignal(str)
    completed = pyqtSignal(object)

    def __init__(self, reference_path: Path, inspection_path: Path, check_rois: list[list[float]], parent: QApplication | None = None) -> None:
        super().__init__(parent)
        self.reference_path = reference_path
        self.inspection_path = inspection_path
        self.check_rois = [list(roi) for roi in check_rois]

    def run(self) -> None:
        try:
            self.stage_changed.emit("正在读取图像")
            source_sha = port_input_sha(self.inspection_path)
            reference_sha = port_input_sha(self.reference_path)
            reference = adaptive.robust.auto.base.read_image(self.reference_path)
            inspection = adaptive.robust.auto.base.read_image(self.inspection_path)

            self.stage_changed.emit("正在对齐图像")
            aligned, alignment = adaptive.robust.auto.automatic_affine(reference, inspection)
            output = adaptive.robust.auto.base.OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
            output.mkdir(parents=True, exist_ok=True)
            if aligned is None:
                report = {
                    "decision": "alignment_uncertain_manual_review",
                    "reference": str(self.reference_path),
                    "inspection": str(self.inspection_path),
                    "alignment": alignment,
                    "alignment_quality": alignment.get("alignment_quality", {"reliable": False, "reason": alignment.get("reason", "unknown")}),
                }
                (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.completed.emit({"status": "alignment_uncertain", "output": str(output), "report": report})
                return

            self.stage_changed.emit("正在生成 DINO 候选")
            overlay, heat, regions = dino_fused_regions(reference, aligned, self.check_rois)
            self.stage_changed.emit("正在准备融合结果")
            adaptive.robust.auto.base.write_image(output / "aligned.jpg", aligned)
            adaptive.robust.auto.base.write_image(output / "dino_anomaly_boxes.jpg", overlay)
            adaptive.robust.auto.base.write_image(output / "anomaly_boxes.jpg", overlay)
            adaptive.robust.auto.base.write_image(output / "check_heatmap.jpg", heat)
            valid_mask = getattr(adaptive.robust.auto, "LAST_WARP_VALID_MASK", None)
            if not isinstance(valid_mask, np.ndarray) or valid_mask.shape != aligned.shape[:2]:
                valid_mask = np.full(aligned.shape[:2], 255, dtype=np.uint8)
            adaptive.robust.auto.base.write_image(output / "valid_warp_mask.png", valid_mask)
            report = {
                "decision": "sam3_fusion_running",
                "reference": str(self.reference_path), "inspection": str(self.inspection_path), "alignment": alignment,
                "alignment_quality": alignment.get("alignment_quality", {"reliable": True, "reason": "legacy_alignment_result"}),
                "local_alignment": robust.LAST_DIAGNOSTICS, "check_region_count": len(self.check_rois),
                "analysis_check_rois": copy.deepcopy(adaptive._merge_rois(self.check_rois)),
                "dino_candidate_count": len(regions), "dino_review_regions": regions,
                "review_regions": regions,
                "image_fingerprints": {
                    "source_sha256": source_sha, "reference_sha256": reference_sha,
                    "stable_during_visual_analysis": source_sha == port_input_sha(self.inspection_path)
                    and reference_sha == port_input_sha(self.reference_path),
                },
                "sam3_fusion": {
                    "status": "running",
                    "policy": "Aligned SAM3 evidence tiers DINO candidates: strong support is green, insufficient support is retained as orange human-review candidates; SAM-led yellow regions also remain human-review only.",
                },
            }
            self.completed.emit({
                "status": "ready_for_sam3",
                "output": str(output),
                "report": report,
                "dino_regions": regions,
            })
        except Exception as error:
            self.completed.emit({"status": "error", "error": str(error).split("\n", 1)[0]})


class Sam3FusionWorker(QThread):
    """Keep CPU-bound SAM3 inference off the Qt event thread."""

    completed = pyqtSignal(object)

    def __init__(
        self,
        reference_path: Path,
        aligned_path: Path,
        valid_mask_path: Path,
        dino_candidates: list[dict[str, Any]],
        heat_path: Path,
        output_dir: Path,
        parent: QApplication | None = None,
    ) -> None:
        super().__init__(parent)
        self.reference_path = reference_path
        self.aligned_path = aligned_path
        self.valid_mask_path = valid_mask_path
        self.dino_candidates = dino_candidates
        self.heat_path = heat_path
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            result = sam3_wire_fusion.fuse_already_aligned(
                reference_path=self.reference_path,
                aligned_inspection_path=self.aligned_path,
                valid_mask_path=self.valid_mask_path,
                dino_candidates=self.dino_candidates,
                dino_heat_path=self.heat_path,
                output_dir=self.output_dir,
            )
        except Exception as error:  # The DINO candidate output remains usable.
            result = {"status": "error", "error": str(error).split("\n", 1)[0]}
        self.completed.emit(result)


class PortCropReviewWorker(QThread):
    """Optional local model work stays off the Qt event thread."""
    completed = pyqtSignal(object)

    def __init__(self, report, output, scene, parent=None):
        super().__init__(parent)
        self.report_snapshot = copy.deepcopy(report)
        self.output = Path(output)
        self.scene = scene

    def run(self):
        try:
            config = self.output / "port_crop_ultralytics_config"
            config.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(config)
            os.environ["YOLO_OFFLINE"] = "True"
            os.environ["YOLO_AUTOINSTALL"] = "False"
            result = run_gui_port_review(self.report_snapshot, project=ROOT, enabled=True, scene=self.scene)
            if result["status"] == "applied":
                render_port_overlay(self.output / "aligned.jpg", result, self.output / "port_crop_hint_boxes.jpg")
            (self.output / "port_crop_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            result = {"status":"fallback", "tile_hints":[], "automatic_fault_verdict":False,
                      "fallback_reason":str(error).split("\n",1)[0]}
        self.completed.emit({"output":str(self.output), "result":result})


class DeepSeekMaskReviewWorker(QThread):
    """Run an explicitly requested external mask review away from the Qt thread."""

    completed = pyqtSignal(object)

    def __init__(
        self,
        reference_mask_path: Path,
        inspection_mask_path: Path,
        candidates: list[dict[str, Any]],
        question_zh: str,
        api_key: str | None = None,
        parent: QApplication | None = None,
    ) -> None:
        super().__init__(parent)
        self.reference_mask_path = reference_mask_path
        self.inspection_mask_path = inspection_mask_path
        self.candidates = candidates
        self.question_zh = question_zh
        self.api_key = api_key

    def run(self) -> None:
        try:
            result = deepseek_mask_review.ask_masks(
                self.reference_mask_path,
                self.inspection_mask_path,
                self.candidates,
                self.question_zh,
                api_key=self.api_key,
            )
        except Exception as error:  # External review is optional and non-authoritative.
            result = {"status": "error", "manual_only": True, "error": str(error).split("\n", 1)[0]}
        self.completed.emit(result)


class DINOReview(adaptive.AdaptiveReview):
    """Single-message review UI: only possible differences are shown."""

    def build(self) -> None:
        super().build()
        self.aligned_view, self.dino_boxes_view, self.dino_heat_view = self.views
        self.sam_difference_view = ZoomImageView("等待 SAM3 差异结果")
        self.sam_detection_view = ZoomImageView("等待 SAM3 线缆检测结果")
        self.fusion_view = ZoomImageView("等待融合复核结果")
        self.tabs.setTabText(0, "对齐待检图")
        self.tabs.setTabText(1, "DINO 差异框")
        self.tabs.setTabText(2, "DINO 差异热图")
        self.tabs.addTab(self.sam_difference_view, "SAM3 差异框")
        self.tabs.addTab(self.sam_detection_view, "SAM3 待检线缆")
        self.tabs.addTab(self.fusion_view, "DINO+SAM3 融合框")
        self.comparison_panel = CandidateComparisonPanel(self)
        self.tabs.addTab(self.comparison_panel, "候选局部对比")
        self.port_hint_view = ZoomImageView("端口提示默认关闭")
        self.tabs.addTab(self.port_hint_view, "端口局部提示")
        self.port_comparison_panel = CandidateComparisonPanel(self)
        self.tabs.addTab(self.port_comparison_panel, "端口局部对比")
        self.port_worker = None
        self._port_visual_report = None
        self.capture_advice_label = QLabel("拍摄建议（本地生成，无需大模型）\n完成检测后显示针对本轮的拍摄建议。", self)
        self.capture_advice_label.setWordWrap(True)
        self.capture_advice_label.setTextFormat(Qt.PlainText)
        self.capture_advice_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_card.layout().addWidget(self.capture_advice_label)
        self.port_hint_switch = QCheckBox("启用可选端口提示", self)
        self.port_hint_switch.setChecked(False)
        self.port_hint_switch.setToolTip("完成原视觉复核后，点击生成；黄色保留原框，蓝色标出端口小框。")
        self.result_card.layout().addWidget(self.port_hint_switch)
        self.port_scene_combo = QComboBox(self)
        self.port_scene_combo.addItem("未知/机柜场景", "unknown")
        self.port_scene_combo.addItem("已确认：现有机箱数据场景", PORT_SCENE)
        self.port_scene_combo.setToolTip("当前模型仅验收了现有机箱数据；此项为操作者确认，非自动识别。")
        self.result_card.layout().addWidget(self.port_scene_combo)
        self.port_hint_button = QPushButton("生成端口局部提示", self)
        self.port_hint_button.setEnabled(False)
        self.port_hint_button.clicked.connect(self.start_port_crop_review)
        self.result_card.layout().addWidget(self.port_hint_button)
        self.port_hint_status = QLabel("端口提示：关闭", self)
        self.port_hint_status.setWordWrap(True)
        self.result_card.layout().addWidget(self.port_hint_status)
        self.port_hint_switch.toggled.connect(self._port_switch_changed)
        self.initial_worker: InitialReviewWorker | None = None
        self.sam3_worker: Sam3FusionWorker | None = None
        self._sam3_pending: dict[str, Any] | None = None
        self.deepseek_worker: DeepSeekMaskReviewWorker | None = None
        self._deepseek_pending: dict[str, Any] | None = None
        self.agent_task_path: Path | None = None
        self._pending_reinspection_task_path: Path | None = None
        self.run_button = next(
            (button for button in self.findChildren(QPushButton) if "自动定位并生成" in button.text()),
            None,
        )
        if self.run_button is not None:
            self.run_button.setText("开始线材差异复核")
        for button in self.findChildren(QPushButton):
            if button.text() in {"添加检查区域", "撤回最后检查区"}:
                button.hide()
        for label in self.findChildren(QLabel):
            if "首次设置" in label.text():
                label.hide()
        controls = self.status.parentWidget().layout()
        stage_row = QWidget(self)
        stage_layout = QHBoxLayout(stage_row)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(7)
        self.busy_indicator = BusyIndicator(stage_row)
        self.stage_label = QLabel("等待检测", stage_row)
        self.stage_label.setStyleSheet("font-weight:bold;color:#374151")
        self.stage_label.setWordWrap(True)
        self.stage_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        stage_layout.addWidget(self.busy_indicator)
        stage_layout.addWidget(self.stage_label, 1)
        controls.insertWidget(controls.indexOf(self.status), stage_row)
        self.status.setWordWrap(True)
        self.status.hide()
        self.deepseek_api_input = QLineEdit(self)
        self.deepseek_api_input.setEchoMode(QLineEdit.Password)
        self.deepseek_api_input.setClearButtonEnabled(True)
        self.deepseek_saved_key_available = self._has_saved_deepseek_api_key()
        self._update_deepseek_api_hint()
        self.result_card.layout().addWidget(self.deepseek_api_input)
        self.deepseek_question_input = QLineEdit(self)
        self.deepseek_question_input.setText("请比较两张掩膜与候选区，说明优先人工检查的位置和可见差异。")
        self.deepseek_question_input.setMaxLength(500)
        self.deepseek_question_input.setToolTip("只会发送此问题、两张二值线缆掩膜和本地候选框；不会发送原图。")
        self.result_card.layout().addWidget(self.deepseek_question_input)
        self.deepseek_review_button = QPushButton("询问 DeepSeek（掩膜）")
        self.deepseek_review_button.setToolTip("手动发送当前问题、两张二值线缆掩膜和本地候选区，获取中文人工复核建议。")
        self.deepseek_review_button.setEnabled(False)
        self.deepseek_review_button.clicked.connect(self.start_deepseek_mask_review)
        self.result_card.layout().addWidget(self.deepseek_review_button)
        self.deepseek_answer_label = QLabel("", self)
        self.deepseek_answer_label.setWordWrap(True)
        self.deepseek_answer_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.deepseek_answer_label.setStyleSheet("color:#374151")
        self.result_card.layout().addWidget(self.deepseek_answer_label)
        self.agent_status_label = QLabel("Agent 工单：等待视觉分析", self)
        self.agent_status_label.setWordWrap(True)
        self.agent_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.agent_status_label.setStyleSheet("font-weight:bold;color:#1f4b7a")
        self.result_card.layout().addWidget(self.agent_status_label)
        self.agent_local_import_button = QPushButton("导入局部复核记录（不确认故障）", self)
        self.agent_local_import_button.setEnabled(False)
        self.agent_local_import_button.clicked.connect(self._import_agent_local_review)
        self.result_card.layout().addWidget(self.agent_local_import_button)
        self.agent_local_summary = QLabel("", self)
        self.agent_local_summary.setWordWrap(True)
        self.agent_local_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.agent_local_summary.setTextFormat(Qt.PlainText)
        self.result_card.layout().addWidget(self.agent_local_summary)
        self.agent_reviewer_input = QLineEdit(self)
        self.agent_reviewer_input.setPlaceholderText("复核人，例如 operator-01")
        self.result_card.layout().addWidget(self.agent_reviewer_input)
        self.agent_outcome_combo = QComboBox(self)
        self.agent_outcome_combo.addItem("确认存在可见差异", "confirmed_difference")
        self.agent_outcome_combo.addItem("确认没有需处理差异", "no_actionable_difference")
        self.agent_outcome_combo.addItem("需要定点补拍", "recapture_required")
        self.result_card.layout().addWidget(self.agent_outcome_combo)
        self.agent_notes_input = QLineEdit(self)
        self.agent_notes_input.setPlaceholderText("人工复核说明（必填）")
        self.result_card.layout().addWidget(self.agent_notes_input)
        self.agent_review_button = QPushButton("记录人工复核到 Agent 工单", self)
        self.agent_review_button.setEnabled(False)
        self.agent_review_button.clicked.connect(self._record_agent_human_review)
        self.result_card.layout().addWidget(self.agent_review_button)
        self.agent_guidance_input = QLineEdit(self)
        self.agent_guidance_input.setPlaceholderText("修改建议；仅在人工确认差异后可记录")
        self.agent_guidance_input.setEnabled(False)
        self.result_card.layout().addWidget(self.agent_guidance_input)
        self.agent_guidance_button = QPushButton("记录修改建议", self)
        self.agent_guidance_button.setEnabled(False)
        self.agent_guidance_button.clicked.connect(self._record_agent_guidance)
        self.result_card.layout().addWidget(self.agent_guidance_button)

        from inspection_agent.workbench_theme import apply_workbench_theme
        apply_workbench_theme(self)
        from inspection_agent.port_rescue_gui import attach_port_rescue
        attach_port_rescue(self, ZoomImageView)

    def _set_stage(self, text: str, busy: bool) -> None:
        self.stage_label.setText(text)
        self.busy_indicator.set_running(busy)
        self.status.setText(text)

    def _set_agent_state(self, state: str, task_path: Path) -> None:
        labels = {
            "awaiting_human_review": "等待人工复核",
            "difference_confirmed": "已确认差异，可填写修改建议",
            "completed_no_actionable_difference": "已完成人工确认：无需处理",
            "recapture_required": "需要定点补拍后复检",
            "repair_guidance_ready": "修改建议已记录，等待复检",
        }
        self.agent_status_label.setText(f"Agent 工单：{labels.get(state, state)}\n{task_path}")
        self.agent_review_button.setEnabled(state == "awaiting_human_review")
        self.agent_local_import_button.setEnabled(state == "awaiting_human_review")
        current = InspectionTask.load(task_path).to_report()
        local = current.get("local_evidence_reviews", [])
        if not local:
            self.agent_local_summary.clear()
        else:
            last = local[-1]
            self.agent_local_summary.setText(f"已保存局部意见 {len(local)} 份；最近记录：{last['case']} / {last['review_mode']}。\n局部支持不是故障确认或连接边。")
        guidance_enabled = state == "difference_confirmed"
        self.agent_guidance_input.setEnabled(guidance_enabled)
        self.agent_guidance_button.setEnabled(guidance_enabled)

    def _create_agent_task(self, report: dict[str, Any]) -> None:
        if self.current_output is None:
            return
        task_path = self._pending_reinspection_task_path or (self.current_output / "agent_task.json")
        try:
            if self._pending_reinspection_task_path is not None:
                task = inspection_agent_bridge.append_reinspection(
                    task_path,
                    inspection=str(report.get("inspection", "")),
                    visual_report=report,
                    source_report=self.current_output / "report.json",
                )
            else:
                task = inspection_agent_bridge.create_task_from_visual_report(
                    task_path,
                    task_id=f"cabinet-{self.current_output.name}",
                    scene_type="cabinet_harness",
                    reference=str(report.get("reference", "")),
                    inspection=str(report.get("inspection", "")),
                    visual_report=report,
                    source_report=self.current_output / "report.json",
                )
        except (OSError, WorkflowError, ValueError) as error:
            if self._pending_reinspection_task_path is None:
                self.agent_task_path = None
            self.agent_status_label.setText(f"Agent 工单未生成：{error}")
            return
        finally:
            self._pending_reinspection_task_path = None
        self.agent_task_path = task_path
        self._set_agent_state(task.state, task_path)

    def _import_agent_local_review(self) -> None:
        if self.agent_task_path is None:
            return
        for name in ("initial_worker", "sam3_worker", "deepseek_worker", "port_worker", "rescue_worker"):
            worker = getattr(self, name, None)
            if worker is not None and worker.isRunning():
                self.agent_local_summary.setText("分析尚在运行，暂不能导入局部记录。")
                return
        task_path = Path(self.agent_task_path)
        try:
            expected = task_fingerprint(task_path)
            packet, _ = QFileDialog.getOpenFileName(self, "选择局部复核导入清单", "", "JSON (*.json)")
            if not packet:
                return
            if self.agent_task_path is None or Path(self.agent_task_path) != task_path:
                raise ValueError("当前工单已更换，导入取消")
            _, summary = import_for_current_view(task_path, Path(packet),
                reference=self.reference.text().strip(), inspection=self.inspection.text().strip(),
                expected_task_sha=expected)
            task = InspectionTask.load(task_path)
            self._set_agent_state(task.state, task_path)
            self.agent_local_summary.setText(summary)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.agent_local_summary.setText(f"局部记录未导入：{error}")

    def _record_agent_human_review(self) -> None:
        if self.agent_task_path is None:
            return
        reviewer = self.agent_reviewer_input.text().strip()
        notes = self.agent_notes_input.text().strip()
        if not reviewer or not notes:
            self.agent_status_label.setText("Agent 工单：请填写复核人和人工复核说明。")
            return
        outcome = str(self.agent_outcome_combo.currentData())
        confirmed_ids: list[str] = []
        if outcome == "confirmed_difference" and self.current_output is not None:
            try:
                visual_report = json.loads((self.current_output / "report.json").read_text(encoding="utf-8"))
                confirmed_ids = [
                    str(item.get("id", index))
                    for index, item in enumerate(visual_report.get("review_regions", []), start=1)
                    if isinstance(item, dict)
                ]
            except (OSError, json.JSONDecodeError):
                confirmed_ids = []
        try:
            task = inspection_agent_bridge.record_human_decision(
                self.agent_task_path,
                reviewer=reviewer,
                outcome=outcome,
                notes=notes,
                confirmed_candidate_ids=confirmed_ids,
            )
        except (OSError, WorkflowError, ValueError) as error:
            self.agent_status_label.setText(f"Agent 人工复核未保存：{error}")
            return
        self._set_agent_state(task.state, self.agent_task_path)

    def _record_agent_guidance(self) -> None:
        if self.agent_task_path is None:
            return
        instruction = self.agent_guidance_input.text().strip()
        if not instruction:
            self.agent_status_label.setText("Agent 工单：请先填写修改建议。")
            return
        try:
            current = InspectionTask.load(self.agent_task_path)
            conclusions = current.to_report()["human_conclusions"]
            evidence_ids = list(conclusions[-1].get("confirmed_candidate_ids", [])) if conclusions else []
            task = inspection_agent_bridge.record_guidance(
                self.agent_task_path,
                instruction=instruction,
                evidence_ids=evidence_ids,
            )
        except (OSError, WorkflowError, ValueError) as error:
            self.agent_status_label.setText(f"Agent 修改建议未保存：{error}")
            return
        self._set_agent_state(task.state, self.agent_task_path)

    @staticmethod
    def _has_saved_deepseek_api_key() -> bool:
        try:
            settings = deepseek_mask_review.load_settings()
            return bool(deepseek_mask_review._local_api_key(settings))
        except deepseek_mask_review.DeepSeekMaskReviewError:
            return False

    def _update_deepseek_api_hint(self) -> None:
        if self.deepseek_saved_key_available:
            self.deepseek_api_input.setPlaceholderText("已保存 API Key（留空即可使用；输入新值可替换）")
            self.deepseek_api_input.setToolTip("已从本项目本地配置读取 API Key。留空即可复核；粘贴新值后点击复核会替换保存值。")
            return
        self.deepseek_api_input.setPlaceholderText("DeepSeek API Key（首次输入后保存）")
        self.deepseek_api_input.setToolTip("首次输入并复核后保存到本项目的本地配置文件；不会写入报告或输出目录。")

    def _refresh_candidate_comparisons(
        self,
        reference_path: Path,
        dino_regions: list[dict[str, Any]],
        sam_components: list[dict[str, Any]] | None = None,
        fusion_regions: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.current_output is None:
            return
        candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(dino_regions, start=1):
            candidates.append({
                **candidate,
                "comparison_key": f"dino_{index:02d}",
                "comparison_label": f"DINO 候选 {index:02d}",
            })
        for index, candidate in enumerate(sam_components or [], start=1):
            direction = "新增" if candidate.get("direction") == "inspection_only" else "缺失"
            candidates.append({
                **candidate,
                "comparison_key": f"sam3_{index:02d}",
                "comparison_label": f"SAM3 {direction} {index:02d}",
            })
        for index, candidate in enumerate(fusion_regions or [], start=1):
            candidates.append({
                **candidate,
                "comparison_key": f"fusion_{index:02d}",
                "comparison_label": f"融合 {candidate.get('id', index)}",
            })
        records = _candidate_comparison_records(
            reference_path,
            self.current_output / "aligned.jpg",
            self.current_output,
            candidates,
        )
        self.comparison_panel.set_records(records)

    def _finish_initial_review(self, result: dict[str, Any]) -> None:
        if self.initial_worker is not None:
            self.initial_worker.deleteLater()
        self.initial_worker = None
        if result.get("status") == "error":
            if self.run_button is not None:
                self.run_button.setEnabled(True)
            self._set_stage("检测失败", False)
            self.set_decision("uncertain", "检测未完成，请人工复核", result.get("error", "unknown error"))
            self._show_capture_advice({"decision": "detection_failed"})
            return

        self.current_output = Path(str(result["output"]))
        report = result["report"]
        if result.get("status") == "alignment_uncertain":
            if self.run_button is not None:
                self.run_button.setEnabled(True)
            self._write_report(report)
            self._create_agent_task(report)
            self._set_stage("定位需要人工复核", False)
            self.set_decision("uncertain", "定位不可靠，请人工复核", "未输出可能差异框。")
            return

        dino_regions = result["dino_regions"]
        self._write_report(report)
        for view, name in zip((self.aligned_view, self.dino_boxes_view, self.dino_heat_view), ("aligned.jpg", "anomaly_boxes.jpg", "check_heatmap.jpg")):
            view.load(self.current_output / name)
        self._refresh_candidate_comparisons(Path(str(report["reference"])), dino_regions)
        self.tabs.setCurrentIndex(1 if dino_regions else 0)
        self.set_decision("processing", "正在检测", "SAM3 正在核验线材证据。")
        self._set_stage("正在进行 SAM3 线缆分析", True)
        worker = Sam3FusionWorker(
            reference_path=Path(str(report["reference"])),
            aligned_path=self.current_output / "aligned.jpg",
            valid_mask_path=self.current_output / "valid_warp_mask.png",
            dino_candidates=dino_regions,
            heat_path=self.current_output / "check_heatmap.jpg",
            output_dir=self.current_output,
            parent=self,
        )
        worker.completed.connect(self._finish_sam3_fusion)
        self.sam3_worker = worker
        self._sam3_pending = {"output": self.current_output, "report": report, "dino_regions": dino_regions}
        worker.start()

    def load_recipe(self) -> dict[str, Any]:
        """Keep the DINO operator UI on its cabinet-only recipe."""
        if not CABINET_RECIPE.exists():
            return {"reference_image": "", "check_rois": []}
        recipe = json.loads(CABINET_RECIPE.read_text(encoding="utf-8"))
        recipe.setdefault("check_rois", [])
        return recipe

    def save_recipe(self) -> None:
        """Persist only the cabinet DINO UI state, never another app's recipe."""
        CABINET_RECIPE.parent.mkdir(parents=True, exist_ok=True)
        CABINET_RECIPE.write_text(
            json.dumps(self.recipe, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_report(self, report: dict[str, Any]) -> None:
        if self.current_output is None:
            return
        report = copy.deepcopy(report)
        report["capture_advice"] = self._show_capture_advice(report)
        (self.current_output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.report.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        self._port_visual_report = copy.deepcopy(report)
        self._update_port_controls()

    def _show_capture_advice(self, report: dict[str, Any]) -> dict[str, Any]:
        advice = build_capture_advice(report)
        self.capture_advice_label.setText(render_capture_advice(advice))
        return advice

    def _update_port_controls(self):
        busy = any(worker is not None and worker.isRunning() for worker in
                   (self.initial_worker, self.sam3_worker, self.deepseek_worker, self.port_worker,
                    getattr(self, "rescue_worker", None)))
        ready = self._port_visual_report is not None and self.current_output is not None and self._port_visual_report.get("decision") in {
            "possible_difference_manual_review", "no_significant_wire_related_difference", "no_significant_difference"}
        if ready:
            ready = (Path(self.reference.text().strip()) == Path(self._port_visual_report.get("reference", ""))
                     and Path(self.inspection.text().strip()) == Path(self._port_visual_report.get("inspection", "")))
        self.port_hint_button.setEnabled(self.port_hint_switch.isChecked() and ready and not busy)
        self.port_hint_switch.setEnabled(not busy)
        self.port_scene_combo.setEnabled(not busy)
        if hasattr(self, "rescue_switch"):
            from inspection_agent.port_rescue_gui import update_port_rescue_controls
            update_port_rescue_controls(self)

    def _port_switch_changed(self, enabled):
        if not enabled:
            self.port_hint_view.scene.clear()
            self.port_hint_view.item = None
            self.port_comparison_panel.set_records([])
            self.port_hint_status.setText("端口提示：关闭")
        else:
            self.port_hint_status.setText("端口提示：开启，完成视觉复核后点击生成")
        self._update_port_controls()

    def start_port_crop_review(self):
        self._update_port_controls()
        if not self.port_hint_button.isEnabled():
            return
        worker = PortCropReviewWorker(self._port_visual_report, self.current_output,
                                      str(self.port_scene_combo.currentData()), self)
        self.port_worker = worker
        worker.completed.connect(self._finish_port_crop_review)
        worker.finished.connect(worker.deleteLater)
        self.port_hint_view.scene.clear()
        self.port_hint_view.item = None
        self.port_comparison_panel.set_records([])
        self.port_hint_status.setText("端口提示：正在检查输入并生成局部框")
        if self.run_button is not None:
            self.run_button.setEnabled(False)
        self.deepseek_review_button.setEnabled(False)
        self._set_stage("正在生成端口局部提示", True)
        worker.start()
        self._update_port_controls()

    def _finish_port_crop_review(self, payload):
        self.port_worker = None
        if self.current_output is None or self.current_output != Path(payload["output"]):
            self._update_port_controls()
            return
        result = payload["result"]
        report = copy.deepcopy(self._port_visual_report)
        report["optional_port_crop"] = {key:value for key,value in result.items()
                                         if key not in {"predictions", "aligned_predictions"}}
        self._write_report(report)
        if result["status"] == "applied":
            self.port_hint_view.load(self.current_output / "port_crop_hint_boxes.jpg")
            candidates = [{**hint["box"], "comparison_key":f"port_{index:02d}",
                           "comparison_label":f"端口提示 {index:02d}（人工复核）"}
                          for index,hint in enumerate(result["tile_hints"],1)]
            self.port_comparison_panel.set_records(_candidate_comparison_records(
                Path(report["reference"]),self.current_output / "aligned.jpg",self.current_output,candidates))
            self.port_hint_status.setText(f"端口提示：新增 {len(candidates)} 个；黄色原复核框，蓝色端口小框。")
            self.tabs.setCurrentWidget(self.port_hint_view)
        else:
            reasons = {"unsupported_scene":"当前场景不在模型验收范围内",
                       "visual_geometry_provenance_missing":"缺少原图指纹或实际配准矩阵，请重新视觉复核",
                       "local_alignment_not_supported":"本轮使用局部形变校正，暂不叠加端口框",
                       "reference_fingerprint_mismatch":"参考图不在当前模型验收范围内",
                       "unreliable_alignment":"配准不可靠"}
            reason = result.get("fallback_reason", "unknown")
            self.port_hint_status.setText("端口提示：保留原结果；" + reasons.get(reason, "输入或模型检查未通过，详见报告"))
        if self.run_button is not None:
            self.run_button.setEnabled(True)
        self.deepseek_review_button.setEnabled(self._deepseek_pending is not None)
        self._set_stage("端口提示处理完成", False)
        self._update_port_controls()

    def closeEvent(self, event):
        if getattr(self, "rescue_worker", None) is not None and self.rescue_worker.isRunning():
            self.rescue_status.setText("补漏正在处理，请完成后关闭窗口。")
            event.ignore()
            return
        if self.port_worker is not None and self.port_worker.isRunning():
            self.port_hint_status.setText("端口提示正在处理，请完成后关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)

    def _finish_sam3_fusion(self, result: dict[str, Any]) -> None:
        pending = self._sam3_pending
        if pending is None or self.current_output != pending["output"]:
            return
        if self.run_button is not None:
            self.run_button.setEnabled(True)
        if self.sam3_worker is not None:
            self.sam3_worker.deleteLater()
        self.sam3_worker = None
        self._sam3_pending = None
        self.deepseek_review_button.setEnabled(False)
        report = pending["report"]
        dino_regions = pending["dino_regions"]
        if result.get("status") != "ok":
            report["decision"] = "possible_difference_manual_review" if dino_regions else "no_significant_difference"
            report["sam3_fusion"] = {
                "status": "error",
                "fallback": "Retained the existing DINO and traditional candidate output.",
                "error": result.get("error", "unknown SAM3 error"),
            }
            report["review_regions"] = dino_regions
            self._write_report(report)
            self._create_agent_task(report)
            if dino_regions:
                self.tabs.setCurrentIndex(1)
                self.set_decision("uncertain", "SAM3 未完成，保留原差异框供人工复核", f"DINO/传统候选：{len(dino_regions)} 个 ｜ SAM3：不可用")
                self._set_stage("SAM3 未完成", False)
            else:
                self.set_decision("clear", "未发现明显差异", "DINO 未生成候选，SAM3 本轮未完成。")
                self._set_stage("SAM3 未完成", False)
            return

        regions = result["review_regions"]
        green = result["green_regions"]
        orange = result["orange_review_regions"]
        yellow = result["yellow_review_regions"]
        report.update(
            decision="possible_difference_manual_review" if regions else "no_significant_wire_related_difference",
            review_regions=regions,
            sam3_fusion=result,
        )
        self._write_report(report)
        self._create_agent_task(report)
        reference_mask = Path(str(result["reference_sam3"]["output_dir"])) / "mask_union.png"
        inspection_mask = Path(str(result["inspection_sam3"]["output_dir"])) / "mask_union.png"
        if regions and reference_mask.is_file() and inspection_mask.is_file():
            self._deepseek_pending = {
                "output": self.current_output,
                "report": report,
                "reference_mask": reference_mask,
                "inspection_mask": inspection_mask,
                "candidates": regions,
            }
            self.deepseek_review_button.setEnabled(True)
        self.sam_difference_view.load(self.current_output / "sam3_difference_boxes.jpg")
        self.sam_detection_view.load(self.current_output / "sam3" / "inspection" / "overlay.jpg")
        self.fusion_view.load(self.current_output / "sam3_fusion_boxes.jpg")
        self._refresh_candidate_comparisons(
            Path(str(report["reference"])),
            dino_regions,
            result["sam_component_audit"],
            regions,
        )
        detail = f"DINO 原候选：{len(dino_regions)} 个 ｜ 绿色融合：{len(green)} 个 ｜ 橙色 DINO-only：{len(orange)} 个 ｜ 黄色弱提示：{len(yellow)} 个"
        if regions:
            self.tabs.setCurrentWidget(self.fusion_view)
            self.set_decision("uncertain", "发现线材相关变化，请人工复核", detail)
            self._set_stage("检测完成", False)
        else:
            self.tabs.setCurrentIndex(0)
            self.set_decision("clear", "未发现需复核的线材相关变化", detail)
            self._set_stage("检测完成", False)

    def start_deepseek_mask_review(self) -> None:
        """Invoke the optional external reviewer only after an operator click."""
        if getattr(self, "rescue_worker", None) is not None and self.rescue_worker.isRunning():
            return
        if self.port_worker is not None and self.port_worker.isRunning():
            return
        pending = self._deepseek_pending
        if pending is None or self.current_output != pending["output"]:
            return
        if self.deepseek_worker is not None and self.deepseek_worker.isRunning():
            return
        api_key = self.deepseek_api_input.text().strip() or None
        question_zh = self.deepseek_question_input.text().strip()
        if not question_zh:
            self._set_stage("请先输入 DeepSeek 问题", False)
            return
        if api_key is not None:
            try:
                deepseek_mask_review.save_local_api_key(deepseek_mask_review.load_settings(), api_key)
            except deepseek_mask_review.DeepSeekMaskReviewError as error:
                self._set_stage(f"DeepSeek API Key 保存失败：{error}", False)
                return
            self.deepseek_api_input.clear()
            self.deepseek_saved_key_available = True
            self._update_deepseek_api_hint()
        self.deepseek_review_button.setEnabled(False)
        if self.run_button is not None:
            self.run_button.setEnabled(False)
        self.deepseek_answer_label.setText("")
        self._set_stage("正在询问 DeepSeek", True)
        worker = DeepSeekMaskReviewWorker(
            reference_mask_path=pending["reference_mask"],
            inspection_mask_path=pending["inspection_mask"],
            candidates=pending["candidates"],
            question_zh=question_zh,
            api_key=api_key,
            parent=self,
        )
        worker.completed.connect(self._finish_deepseek_mask_review)
        self.deepseek_worker = worker
        worker.start()

    def _finish_deepseek_mask_review(self, result: dict[str, Any]) -> None:
        pending = self._deepseek_pending
        if pending is None or self.current_output != pending["output"]:
            return
        if self.deepseek_worker is not None:
            self.deepseek_worker.deleteLater()
        self.deepseek_worker = None
        if self.run_button is not None:
            self.run_button.setEnabled(True)
        self.deepseek_review_button.setEnabled(True)
        report = pending["report"]
        # This appends a non-fatal external-review record. Local regions and the
        # local decision are intentionally never overwritten by this branch.
        report["external_mask_review"] = result
        self._write_report(report)
        if result.get("status") == "ok":
            self.deepseek_answer_label.setText("DeepSeek：\n" + str(result.get("answer_zh", "")))
            suffix = "（已自动补发一次）" if result.get("answer_retry_count") else ""
            self._set_stage("DeepSeek 回答已返回" + suffix, False)
        else:
            error = str(result.get("error", "unknown error"))
            if error.startswith("DeepSeek response is not valid JSON"):
                error = "DeepSeek 服务返回的不是 JSON。\n" + error[len("DeepSeek response is not valid JSON"):].lstrip()
            elif error.startswith("DeepSeek model did not return the required JSON object"):
                error = "DeepSeek 模型未按要求返回 JSON 排序结果。"
            self.deepseek_answer_label.setText("")
            self._set_stage("DeepSeek 复核未完成：\n" + error, False)

    def run(self) -> None:
        if getattr(self, "rescue_worker", None) is not None and self.rescue_worker.isRunning():
            return
        self._rescue_token = None
        if hasattr(self, "rescue_view"):
            self.rescue_view.scene.clear()
            self.rescue_view.item = None
        if self.port_worker is not None and self.port_worker.isRunning():
            return
        if self.initial_worker is not None and self.initial_worker.isRunning():
            return
        if self.sam3_worker is not None and self.sam3_worker.isRunning():
            return
        if self.deepseek_worker is not None and self.deepseek_worker.isRunning():
            return
        self._deepseek_pending = None
        self.capture_advice_label.setText("拍摄建议（本地生成，无需大模型）\n等待本轮检测结果；建议与参考图保持相同角度和取景范围。")
        self._port_visual_report = None
        self.port_hint_view.scene.clear()
        self.port_hint_view.item = None
        self.port_comparison_panel.set_records([])
        self.port_hint_button.setEnabled(False)
        self.deepseek_review_button.setEnabled(False)
        self.comparison_panel.set_records([])
        reference_path, inspection_path = Path(self.reference.text().strip()), Path(self.inspection.text().strip())
        if not reference_path.is_file() or not inspection_path.is_file():
            self._set_stage("等待输入", False)
            self.set_decision("idle", "尚未检测", "请选择正确参考图和待检图。")
            return
        check_rois = self.recipe.get("check_rois") or [[0.0, 0.0, 1.0, 1.0]]
        continuing_task: Path | None = None
        if self.agent_task_path is not None and self.agent_task_path.is_file():
            try:
                existing_task = InspectionTask.load(self.agent_task_path)
                if existing_task.state in {"repair_guidance_ready", "recapture_required"}:
                    continuing_task = self.agent_task_path
            except (OSError, WorkflowError, ValueError):
                continuing_task = None
        self._pending_reinspection_task_path = continuing_task
        if continuing_task is None:
            self.agent_task_path = None
        self.agent_review_button.setEnabled(False)
        self.agent_local_import_button.setEnabled(False)
        self.agent_local_summary.clear()
        self.agent_guidance_input.setEnabled(False)
        self.agent_guidance_button.setEnabled(False)
        if continuing_task is None:
            self.agent_status_label.setText("Agent 工单：等待本轮视觉分析完成")
        else:
            self.agent_status_label.setText(f"Agent 工单：正在执行复检\n{continuing_task}")
        self.current_output = None
        self.set_decision("processing", "正在检测", "正在准备图像分析。")
        self._set_stage("正在读取图像", True)
        if self.run_button is not None:
            self.run_button.setEnabled(False)
        worker = InitialReviewWorker(
            reference_path=reference_path,
            inspection_path=inspection_path,
            check_rois=check_rois,
            parent=self,
        )
        worker.stage_changed.connect(lambda text: self._set_stage(text, True))
        worker.completed.connect(self._finish_initial_review)
        self.initial_worker = worker
        worker.start()
        self._update_port_controls()


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = DINOReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
