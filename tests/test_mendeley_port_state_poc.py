from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "prototype"))

from prepare_mendeley_port_state_poc import convert_label_text  # noqa: E402
from mendeley_port_state_poc import build_report  # noqa: E402
from evaluate_mendeley_port_state_poc import greedy_match, iou  # noqa: E402


class MendeleyPortStatePocTests(unittest.TestCase):
    def test_only_unplugged_source_classes_are_retained_as_rectangles(self) -> None:
        converted = convert_label_text(
            "1 0.5 0.5 0.1 0.1\n"
            "3 0.5 0.5 0.2 0.4\n"
            "4 0.2 0.3 0.1 0.2\n"
            "2 0.1 0.2 0.1 0.1\n"
        )
        self.assertEqual(len(converted), 2)
        self.assertTrue(converted[0].startswith("0 "))
        self.assertTrue(converted[1].startswith("1 "))
        self.assertEqual(len(converted[0].split()), 9)

    def test_empty_candidate_result_never_claims_ok(self) -> None:
        report = build_report(Path("image.jpg"), Path("weights.pt"), 0.25, [])
        self.assertEqual(report["decision"], "no_unseated_port_candidate_not_verified")

    def test_candidate_result_requires_manual_review(self) -> None:
        records = [{"class": "unplugged_plug", "confidence": 0.9, "box_xyxy": [1, 2, 3, 4]}]
        report = build_report(Path("image.jpg"), Path("weights.pt"), 0.25, records)
        self.assertEqual(report["decision"], "possible_unseated_port_manual_review")

    def test_matching_requires_class_and_overlap(self) -> None:
        truth = [{"class_id": 0, "box_xyxy": [0, 0, 20, 20]}]
        predictions = [
            {"class_id": 1, "box_xyxy": [0, 0, 20, 20]},
            {"class_id": 0, "box_xyxy": [10, 10, 30, 30]},
        ]
        self.assertAlmostEqual(iou(truth[0]["box_xyxy"], predictions[1]["box_xyxy"]), 1 / 7)
        matches, missed, extra = greedy_match(truth, predictions, threshold=0.1)
        self.assertEqual(matches, [(0, 1)])
        self.assertEqual(missed, [])
        self.assertEqual(extra, [0])


if __name__ == "__main__":
    unittest.main()
