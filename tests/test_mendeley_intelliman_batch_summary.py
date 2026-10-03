from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assembly_engine.mendeley_intelliman_batch_summary import (
    MendeleyIntelliManBatchSummaryError,
    summarize_mendeley_intelliman_evidence_reports,
)


def report(image_label: str, labels: list[tuple[int, bool]]) -> dict:
    return {
        "schema_version": 1,
        "manual_evaluation_only": True,
        "source": {"dataset_image_label": image_label},
        "label_evidence": [
            {"source_class_id": class_id, "topology_evidence_present": evidence_present}
            for class_id, evidence_present in labels
        ],
    }


class MendeleyIntelliManBatchSummaryTests(unittest.TestCase):
    def test_summary_keeps_source_classes_opaque_and_groups_by_filename_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "damaged.json", root / "disconnected.json"
            first.write_text(json.dumps(report("damaged", [(1, False), (2, True)])), encoding="utf-8")
            second.write_text(json.dumps(report("disconnected", [(1, True)])), encoding="utf-8")
            summary = summarize_mendeley_intelliman_evidence_reports([first, second])
        self.assertEqual(summary["overall"]["source_label_count"], 3)
        self.assertEqual(summary["overall"]["labels_with_topology_evidence"], 2)
        self.assertEqual(summary["overall"]["topology_evidence_coverage_ratio"], 0.666667)
        self.assertEqual(summary["by_dataset_image_label"]["damaged"]["source_label_count"], 2)
        self.assertEqual(summary["by_opaque_source_class_id"]["1"]["report_count"], 2)
        self.assertEqual(summary["by_opaque_source_class_id"]["1"]["source_label_count"], 2)
        self.assertIn("fault_class_accuracy", summary["not_claimed"])

    def test_prediction_like_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_manual.json"
            path.write_text(json.dumps({"schema_version": 1, "manual_evaluation_only": False}), encoding="utf-8")
            with self.assertRaisesRegex(MendeleyIntelliManBatchSummaryError, "not manual-evaluation-only"):
                summarize_mendeley_intelliman_evidence_reports([path])


if __name__ == "__main__":
    unittest.main()
