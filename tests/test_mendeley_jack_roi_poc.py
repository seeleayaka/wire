from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "prototype"))

from evaluate_mendeley_jack_roi_classifier import binary_metrics  # noqa: E402
from mendeley_jack_roi_poc import build_report  # noqa: E402
from prepare_mendeley_jack_roi_dataset import expanded_crop_bounds, grid_rois, roi_contains_empty_jack, read_source_boxes  # noqa: E402
from train_mendeley_jack_roi_classifier import choose_threshold  # noqa: E402


class MendeleyJackRoiPocTests(unittest.TestCase):
    def test_roi_assignment_uses_box_centres(self) -> None:
        roi = [0.2, 0.3, 0.4, 0.6]
        self.assertTrue(roi_contains_empty_jack(roi, [(0.2, 0.3, 0.1, 0.1)]))
        self.assertFalse(roi_contains_empty_jack(roi, [(0.401, 0.6, 0.1, 0.1)]))

    def test_context_crop_is_clipped(self) -> None:
        self.assertEqual(expanded_crop_bounds([0.0, 0.0, 0.2, 0.2], 3.0), (0.0, 0.0, 0.4, 0.4))

    def test_source_class_reader_can_select_disconnected_plug(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "labels.txt"
            path.write_text("3 0.2 0.2 0.1 0.1\n4 0.7 0.7 0.1 0.1\n", encoding="utf-8")
            self.assertEqual(read_source_boxes(path, 3), [(0.2, 0.2, 0.1, 0.1)])

    def test_grid_rois_are_non_overlapping_and_cover_the_frame(self) -> None:
        rois = grid_rois(4, 3)
        self.assertEqual(len(rois), 12)
        self.assertEqual(rois[0]["roi_normalized_xyxy"], [0.0, 0.0, 0.25, 1 / 3])
        self.assertEqual(rois[-1]["roi_normalized_xyxy"], [0.75, 2 / 3, 1.0, 1.0])

    def test_validation_threshold_prefers_empty_jack_signal(self) -> None:
        threshold, metrics = choose_threshold(np_array([0, 0, 1, 1]), np_array([0.1, 0.2, 0.8, 0.9]))
        self.assertGreater(metrics["f1"], 0.99)
        self.assertGreater(threshold, 0.2)

    def test_binary_metrics_count_false_negatives(self) -> None:
        metrics = binary_metrics(np_array([0, 1, 1]), np_array([False, True, False]))
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["fp"], 0)

    def test_no_candidate_never_claims_connector_is_seated(self) -> None:
        report = build_report(Path("image.JPG"), Path("weights.pt"), 0.3, [])
        self.assertEqual(report["decision"], "no_visible_empty_jack_candidate_not_verified")

    def test_plug_report_has_a_distinct_candidate_decision(self) -> None:
        report = build_report(Path("image.JPG"), Path("weights.pt"), 0.6, [{"roi_id": "grid_r01_c01"}], "visible_disconnected_plug")
        self.assertEqual(report["decision"], "possible_visible_disconnected_plug_manual_review")


def np_array(values: list[float | int | bool]):
    import numpy as np

    return np.asarray(values)


if __name__ == "__main__":
    unittest.main()
