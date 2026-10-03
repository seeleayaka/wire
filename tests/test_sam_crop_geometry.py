from copy import deepcopy
import unittest
import sys
sys.path.insert(0, "E:/PythonProject10")
try:
    from inspection_agent.sam_crop_geometry import upper_right_half_box, translate_crop_record
except ModuleNotFoundError:
    from sam_crop_geometry import upper_right_half_box, translate_crop_record


def record():
    return {"geometry_contract_version": 1, "geometry_pair_eligible": True,
        "endpoint_status": "unbranched_two_tip_visible_segment", "candidate_tip_count": 2,
        "branch_cluster_count": 0, "skeleton_component_count": 1,
        "candidate_tips_xy": [[10, 10], [90, 20]], "visible_ends_xy": [[10, 10], [90, 20]]}


class CropGeometryTests(unittest.TestCase):
    def test_integer_quadrant_handles_odd_sizes(self):
        self.assertEqual(upper_right_half_box([575, 517]), [287, 0, 575, 258])
        self.assertEqual(upper_right_half_box([620, 622]), [310, 0, 620, 311])

    def test_translation_preserves_full_geometry(self):
        source = record()
        before = deepcopy(source)
        result = translate_crop_record(source, [100, 50, 200, 110], [300, 200], [10, 10, 91, 21])
        self.assertEqual(result["visible_ends_xy"], [[110, 60], [190, 70]])
        self.assertTrue(result["geometry_pair_eligible"])
        self.assertEqual(source, before)

    def test_every_boundary_blocks_pair(self):
        for box in [[0, 10, 91, 21], [10, 0, 91, 21], [10, 10, 100, 21], [10, 10, 91, 60]]:
            with self.subTest(box=box):
                result = translate_crop_record(record(), [100, 50, 200, 110], [300, 200], box)
                self.assertFalse(result["geometry_pair_eligible"])
                self.assertEqual(result["visible_ends_xy"], [])
                self.assertEqual(result["pre_crop_guard_geometry"]["visible_ends_xy"], [[110, 60], [190, 70]])

    def test_guard_whole_component_not_only_tips(self):
        result = translate_crop_record(record(), [100, 50, 200, 110], [300, 200], [0, 5, 99, 22])
        self.assertTrue(result["crop_evidence"]["boundary_truncated"])
        self.assertFalse(result["geometry_pair_eligible"])

    def test_crop_truncation_cannot_emit_topology_edge(self):
        from inspection_agent.sam_topology_adapter import connection_graph_from_sam_endpoints
        source = record()
        source.update(record_id="constructed_crop_segment", source_score=.95)
        shifted = translate_crop_record(source, [100, 50, 200, 110], [300, 200], [0, 5, 99, 22])
        ports = {"schema_version": 1, "graph_id": "constructed_test", "scene_type": "constructed",
            "terminals": [{"id": "A", "bbox_xyxy": [100, 50, 125, 75]},
                          {"id": "B", "bbox_xyxy": [180, 50, 200, 85]}]}
        graph, audit = connection_graph_from_sam_endpoints({"geometry_contract_version": 1, "records": [shifted]}, ports)
        self.assertEqual(graph.connections, ())
        self.assertEqual(audit["summary"]["candidate_record_count"], 0)

    def test_ambiguous_original_geometry_stays_ambiguous(self):
        source = record()
        source.update(geometry_pair_eligible=False, branch_cluster_count=1, visible_ends_xy=[])
        result = translate_crop_record(source, [100, 50, 200, 110], [300, 200], [10, 10, 91, 21])
        self.assertFalse(result["geometry_pair_eligible"])
        self.assertEqual(result["visible_ends_xy"], [])

    def test_invalid_offsets_points_and_margin_rejected(self):
        for crop in [[100, 50, 400, 110], [200, 50, 100, 110], [0.0, 50, 100, 110]]:
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                translate_crop_record(record(), crop, [300, 200], [10, 10, 91, 21])
        with self.assertRaises(ValueError):
            translate_crop_record(record(), [100, 50, 200, 110], [300, 200], [10, 10, 91, 21], -1)
        source = record()
        source["visible_ends_xy"] = [[110, 10], [90, 20]]
        with self.assertRaises(ValueError):
            translate_crop_record(source, [100, 50, 200, 110], [300, 200], [10, 10, 91, 21])


if __name__ == "__main__":
    unittest.main()
