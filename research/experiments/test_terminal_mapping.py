from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import numpy as np

from inspection_agent.terminal_mapping import create_mapping_draft, review_mapped_topology, validate_mapping
from inspection_agent.visible_segment_geometry import assess_visible_skeleton


class TerminalMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "constructed_contract_fixture.png"
        Image.new("RGB", (100, 60), "white").save(self.image)
        self.mapping = create_mapping_draft(self.image, "controlled_fixture")
        self.mapping["ports"] = [
            {"id": name, "device_id": "fixture_device", "terminal_label": name,
             "roi_kind": "wire_entry_port", "bbox_xyxy": box, "confirmed": True,
             "reviewer": "fixture_author_not_field_operator", "evidence_note": "constructed test port"}
            for name, box in [("A", [0, 0, 20, 20]), ("B", [80, 0, 100, 20])]
        ]
        self.mapping["expected_connections"] = [{"id": "edge1", "from": "A", "to": "B"}]
        self.mapping["expected_review"] = {"confirmed": True, "reviewer": "fixture_author",
            "evidence_note": "declared synthetic relation, not physical wiring truth"}
        self.mapping["scope"] = {"description": "constructed two-port scope", "expected_complete": True}
        skeleton = np.zeros((60, 100), dtype=np.uint8)
        skeleton[10, 10:91] = 1
        record = assess_visible_skeleton(skeleton)
        record.update(record_id="constructed_segment_1", source_score=.9)
        self.endpoint = {"image_binding": deepcopy(self.mapping["image_binding"]), "geometry_contract_version": 1,
                         "records": [record], "coverage_review": {"complete": True,
                         "reviewer": "fixture_author", "evidence_note": "constructed closed test scene",
                         "map_id": self.mapping["map_id"]}}

    def review(self, mapping=None, endpoint=None):
        return review_mapped_topology(mapping or self.mapping, self.image, endpoint or self.endpoint)

    def test_draft_never_copies_bodies_to_ports(self):
        inventory = {"source": {"image_sha256": self.mapping["image_binding"]["image_sha256"]},
            "image_size": [100, 60], "objects": [{"id": "body1", "bbox_xyxy": [0, 0, 100, 60]}]}
        draft = create_mapping_draft(self.image, "other_scene", inventory)
        self.assertEqual(draft["ports"], [])
        self.assertIsNone(draft["expected_connections"])
        result = review_mapped_topology(draft, self.image)
        self.assertEqual(result["decision"], "insufficient_evidence")

    def test_declared_agreement_is_manual_review_not_fault_or_all_clear(self):
        result = self.review()
        self.assertEqual(result["decision"], "declared_visible_topology_agreement_manual_review")
        self.assertFalse(result["automatic_fault_verdict"])
        self.assertEqual(result["adapter_audit"]["summary"]["emitted_connection_count"], 1)
        self.assertEqual(result["raw_comparison"]["decision"], "match")

    def test_draft_ports_cannot_emit_edges(self):
        self.mapping["ports"][0]["confirmed"] = False
        result = self.review()
        self.assertIsNone(result["observed_graph"])
        self.assertEqual(result["decision"], "insufficient_evidence")

    def test_body_roi_is_rejected(self):
        self.mapping["ports"][0]["roi_kind"] = "terminal_body"
        with self.assertRaises(ValueError):
            self.review()

    def test_map_and_endpoint_source_mismatches_are_rejected(self):
        for target in ("map", "endpoint"):
            for key, value in [("image_sha256", "0" * 64), ("image_size", [200, 120]),
                               ("coordinate_frame", "aligned_reference_pixels")]:
                mapping, endpoint = deepcopy(self.mapping), deepcopy(self.endpoint)
                (mapping if target == "map" else endpoint)["image_binding"][key] = value
                with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                    self.review(mapping, endpoint)

    def test_changed_image_bytes_are_rejected(self):
        Image.new("RGB", (100, 60), "black").save(self.image)
        with self.assertRaises(ValueError):
            self.review()

    def test_unknown_expected_table_does_not_become_empty_match(self):
        self.mapping["expected_connections"] = None
        self.mapping["expected_review"]["confirmed"] = False
        result = self.review()
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertIsNone(result["raw_comparison"])

    def test_empty_evidence_never_proves_absence(self):
        self.mapping["expected_connections"] = []
        self.endpoint["records"] = []
        result = self.review()
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertIn("no_visible_segment_records", result["reasons"])

    def test_missing_coverage_stays_insufficient(self):
        self.endpoint.pop("coverage_review")
        result = self.review()
        self.assertEqual(result["raw_comparison"]["decision"], "match")
        self.assertEqual(result["decision"], "insufficient_evidence")

    def test_hidden_or_ambiguous_end_is_not_an_automatic_missing_wire(self):
        self.endpoint["records"][0]["visible_ends_xy"] = [[10, 10], [50, 40]]
        self.endpoint["records"][0]["candidate_tips_xy"] = [[10, 10], [50, 40]]
        result = self.review()
        self.assertEqual(result["raw_comparison"]["decision"], "mismatch_manual_confirmation_required")
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertIn("unresolved_visible_segment_endpoints", result["reasons"])

    def test_overlapping_port_boxes_do_not_assign_by_nearest(self):
        self.mapping["ports"][1]["bbox_xyxy"] = [0, 0, 100, 20]
        result = self.review()
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(result["observed_graph"]["connections"], [])

    def test_fixed_low_score_gate(self):
        self.endpoint["records"][0]["source_score"] = .74
        self.assertEqual(self.review()["decision"], "insufficient_evidence")
        self.endpoint["records"][0]["source_score"] = .75
        self.assertEqual(self.review()["decision"], "declared_visible_topology_agreement_manual_review")

    def test_geometry_contract_required(self):
        self.endpoint.pop("geometry_contract_version")
        result = self.review()
        self.assertIsNone(result["observed_graph"])
        self.assertEqual(result["decision"], "insufficient_evidence")

    def test_branched_geometry_does_not_make_edge(self):
        self.endpoint["records"][0]["branch_cluster_count"] = 1
        self.assertEqual(self.review()["observed_graph"]["connections"], [])

    def test_invalid_roi_confirmation_and_numbers_rejected(self):
        for mutate in [lambda m: m["ports"][0].update(bbox_xyxy=[0, 0, float("nan"), 20]),
                       lambda m: m["ports"][0].update(bbox_xyxy=[0, 0, 110, 20]),
                       lambda m: m["ports"][0].update(confirmed=1),
                       lambda m: m["ports"][0].update(reviewer=""),
                       lambda m: m["ports"][1].update(id="A")]:
            mapping = deepcopy(self.mapping)
            mutate(mapping)
            with self.assertRaises(ValueError):
                validate_mapping(mapping, self.image)
        self.endpoint["records"][0]["source_score"] = float("nan")
        with self.assertRaises(ValueError):
            self.review()

    def test_same_algorithm_new_ids_and_scaled_coordinates(self):
        self.image = Path(self.temp.name) / "second_constructed_fixture.png"
        Image.new("RGB", (200, 120), "white").save(self.image)
        new = create_mapping_draft(self.image, "different_declared_scene")
        self.mapping["map_id"], self.mapping["scene_type"] = new["map_id"], new["scene_type"]
        self.mapping["image_binding"] = new["image_binding"]
        for port, name in zip(self.mapping["ports"], ["cabinet_B:X7:1", "cabinet_B:K4:2"]):
            port["id"] = name
            port["bbox_xyxy"] = [x * 2 for x in port["bbox_xyxy"]]
        self.mapping["expected_connections"][0].update({"from": self.mapping["ports"][0]["id"], "to": self.mapping["ports"][1]["id"]})
        self.endpoint["image_binding"] = deepcopy(new["image_binding"])
        self.endpoint["coverage_review"]["map_id"] = new["map_id"]
        skeleton = np.zeros((120, 200), dtype=np.uint8)
        skeleton[20, 20:181] = 1
        self.endpoint["records"] = [dict(assess_visible_skeleton(skeleton), record_id="scaled_constructed_segment", source_score=.9)]
        self.assertEqual(self.review()["decision"], "declared_visible_topology_agreement_manual_review")


if __name__ == "__main__":
    unittest.main()
