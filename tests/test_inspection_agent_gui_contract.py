from __future__ import annotations

import os
from pathlib import Path
import sys
import json
import tempfile
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import assembly_auto_review_dino as review_app  # noqa: E402


class InspectionAgentGuiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = review_app.QApplication.instance() or review_app.QApplication([])

    def test_agent_controls_exist_and_start_disabled(self) -> None:
        window = review_app.DINOReview()
        try:
            self.assertEqual(window.agent_status_label.text(), "Agent 工单：等待视觉分析")
            self.assertFalse(window.agent_review_button.isEnabled())
            self.assertFalse(window.agent_guidance_button.isEnabled())
            self.assertEqual(window.agent_outcome_combo.count(), 3)
        finally:
            window.close()
            self.application.processEvents()

    def test_gui_creates_task_and_records_human_clear_conclusion(self) -> None:
        window = review_app.DINOReview()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                report = {
                    "decision": "no_significant_wire_related_difference",
                    "reference": "right.png",
                    "inspection": "same.png",
                    "alignment_quality": {"reliable": True},
                    "review_regions": [],
                }
                (output / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False), encoding="utf-8"
                )
                window.current_output = output
                window._create_agent_task(report)
                self.assertTrue((output / "agent_task.json").is_file())
                self.assertTrue(window.agent_review_button.isEnabled())
                window.agent_reviewer_input.setText("operator-test")
                window.agent_outcome_combo.setCurrentIndex(1)
                window.agent_notes_input.setText("人工确认没有需处理差异")
                window._record_agent_human_review()
                saved = json.loads((output / "agent_task.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["state"], "completed_no_actionable_difference")
                self.assertFalse(window.agent_review_button.isEnabled())
        finally:
            window.close()
            self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
