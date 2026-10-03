from __future__ import annotations

import unittest

from tools.audit_mendeley_candidate_postprocess import iou, select, summarize


class CandidatePostprocessTests(unittest.TestCase):
    def test_iou_is_zero_for_disjoint_boxes(self) -> None:
        self.assertEqual(iou({"left": 0, "top": 0, "right": 10, "bottom": 10}, [20, 20, 30, 30]), 0.0)

    def test_original_limit_keeps_existing_order(self) -> None:
        candidates = [
            {"left": 0, "top": 0, "right": 10, "bottom": 10, "area": 100, "difference_score": 1},
            {"left": 20, "top": 20, "right": 30, "bottom": 30, "area": 100, "difference_score": 10},
        ]
        self.assertEqual(select(candidates, ranking="original", limit=1, scale=1.0), candidates[:1])

    def test_summary_tracks_box_and_candidate_hits_separately(self) -> None:
        candidates = [
            {"left": 0, "top": 0, "right": 10, "bottom": 10, "area": 100, "difference_score": 10},
            {"left": 30, "top": 30, "right": 40, "bottom": 40, "area": 100, "difference_score": 5},
        ]
        cases = [{"candidates": candidates, "target_boxes_aligned_xyxy": [[0, 0, 10, 10], [15, 15, 20, 20]]}]
        result = summarize(cases, ranking="original", limit=2, scale=1.0)
        self.assertEqual(result["fault_images_with_target_overlap"], 1)
        self.assertEqual(result["target_boxes_with_overlap"], 1)
        self.assertEqual(result["candidates_overlapping_target"], 1)
        self.assertEqual(result["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
