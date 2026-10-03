from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "manual_review"))

import sam3_visible_endpoint_rule_engine as endpoint_rules  # noqa: E402


class Sam3VisibleEndpointRuleEngineTests(unittest.TestCase):
    def test_endpoint_hits_are_attached_to_existing_candidate_without_new_boxes(self) -> None:
        reference = [{"record_id": "r_01", "visible_ends_xy": [[12, 12], [45, 45]], "endpoint_status": "unclassified_visible_segment_end"}]
        inspection = [{"record_id": "i_01", "visible_ends_xy": [[80, 80]], "endpoint_status": "unclassified_visible_segment_end"}]
        candidates = [{"candidate_id": "candidate_001", "bbox_xyxy": [10, 10, 25, 25], "direction": "reference_only", "local_evidence": {"component_pixels": 90}}]
        evidence = endpoint_rules.build_endpoint_evidence(reference, inspection, candidates, padding=2)
        item = evidence["candidates"][0]
        self.assertEqual(item["bbox_xyxy"], [10, 10, 25, 25])
        self.assertEqual(item["reference_visible_segment_endpoints"][0]["record_id"], "r_01")
        self.assertEqual(item["reference_visible_segment_endpoints"][0]["endpoint_hits"][0]["xy"], [12, 12])
        self.assertEqual(item["inspection_visible_segment_endpoints"], [])
        self.assertEqual(item["endpoint_rule_state"], "visible_segment_endpoints_near_candidate_manual_review")

    def test_no_endpoint_still_requires_manual_review_not_a_fault_label(self) -> None:
        evidence = endpoint_rules.build_endpoint_evidence([], [], [{"candidate_id": "candidate_001", "bbox_xyxy": [10, 10, 25, 25], "direction": None, "local_evidence": {}}], padding=0)
        self.assertEqual(evidence["candidates"][0]["endpoint_rule_state"], "no_visible_segment_endpoint_near_candidate_manual_review")

    def test_mainline_contract_rejects_unrelated_pre_alignment_endpoint_reports(self) -> None:
        mainline = {
            "alignment_quality": {"reliable": True},
            "sam3_fusion": {
                "status": "ok",
                "reference_sam3": {"output_dir": "C:/case/reference"},
                "inspection_sam3": {"output_dir": "C:/case/aligned-inspection"},
            },
        }
        with self.assertRaisesRegex(endpoint_rules.EndpointRuleError, "reference endpoint report"):
            endpoint_rules.validate_mainline_contract(
                mainline,
                {"source_dir": "C:/case/wrong-reference"},
                {"source_dir": "C:/case/aligned-inspection"},
            )

    def test_candidate_records_accept_mainline_review_regions(self) -> None:
        records = endpoint_rules.candidate_records({"review_regions": [{"id": "green_01", "bbox_xyxy": [1, 2, 8, 9], "difference_score": 3.5, "sam_support_pixels": 72}]})
        self.assertEqual(records[0]["candidate_id"], "green_01")
        self.assertEqual(records[0]["bbox_xyxy"], [1, 2, 8, 9])
        self.assertEqual(records[0]["local_evidence"], {"difference_score": 3.5, "sam_support_pixels": 72})


if __name__ == "__main__":
    unittest.main()
