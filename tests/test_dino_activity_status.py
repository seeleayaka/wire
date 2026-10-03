from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import assembly_auto_review_dino as dino  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


class DINOActivityStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_stage_indicator_tracks_worker_activity(self) -> None:
        window = dino.DINOReview()
        try:
            self.assertEqual(window.run_button.text(), "开始线材差异复核")
            self.assertEqual(
                [window.tabs.tabText(index) for index in range(window.tabs.count())],
                ["对齐待检图", "DINO 差异框", "DINO 差异热图", "SAM3 差异框", "SAM3 待检线缆", "DINO+SAM3 融合框", "候选局部对比",
                 "端口局部提示", "端口局部对比", "独立端口补漏"],
            )
            # Existing optional panels remain visible, but model requests start off.
            self.assertFalse(window.rescue_switch.isChecked())
            self.assertFalse(window.rescue_supplement_switch.isChecked())
            self.assertFalse(window.rescue_student_switch.isChecked())
            self.assertFalse(window.rescue_feature_switch.isChecked())
            self.assertEqual(window.deepseek_api_input.echoMode(), dino.QLineEdit.Password)
            expected_hint = "已保存 API Key" if window.deepseek_saved_key_available else "首次输入后保存"
            self.assertIn(expected_hint, window.deepseek_api_input.placeholderText())
            self.assertTrue(window.stage_label.wordWrap())
            hidden_controls = {
                button.text(): button.isHidden()
                for button in window.findChildren(dino.QPushButton)
                if button.text() in {"添加检查区域", "撤回最后检查区"}
            }
            self.assertEqual(hidden_controls, {"添加检查区域": True, "撤回最后检查区": True})
            window._set_stage("正在对齐图像", True)
            self.assertEqual(window.stage_label.text(), "正在对齐图像")
            self.assertTrue(window.busy_indicator._timer.isActive())

            window._set_stage("检测完成", False)
            self.assertEqual(window.stage_label.text(), "检测完成")
            self.assertFalse(window.busy_indicator._timer.isActive())
            self.assertTrue(window.busy_indicator.isHidden())
        finally:
            window.close()

    def test_candidate_crop_bounds_accepts_both_candidate_contracts(self) -> None:
        self.assertEqual(
            dino._candidate_crop_bounds(
                {"left": 10, "top": 20, "right": 30, "bottom": 50}, 100, 100
            ),
            (0, 0, 54, 74, (10, 20, 30, 50)),
        )
        self.assertEqual(
            dino._candidate_crop_bounds(
                {"bbox_xyxy": [80, 85, 100, 100]}, 100, 100
            ),
            (56, 61, 100, 100, (80, 85, 100, 100)),
        )

    def test_deepseek_worker_forwards_one_time_ui_key(self) -> None:
        received: dict[str, object] = {}

        def fake_review(*args: object, **kwargs: object) -> dict[str, object]:
            received["api_key"] = kwargs.get("api_key")
            return {"status": "ok"}

        worker = dino.DeepSeekMaskReviewWorker(
            Path("reference.png"), Path("inspection.png"), [], "问题", api_key="one-time-key"
        )
        with patch.object(dino.deepseek_mask_review, "ask_masks", side_effect=fake_review):
            worker.run()
        self.assertEqual(received["api_key"], "one-time-key")


if __name__ == "__main__":
    unittest.main()
