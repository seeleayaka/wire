from copy import deepcopy
import itertools
import unittest

import cv2
import numpy as np

from core import extract_mask, trace_skeleton, assess_view, compare_views, validate_ports


REVIEW = {"confirmed": True, "reviewer": "synthetic-fixture",
          "evidence_note": "Constructed software test, not a real photo annotation."}
PENDING = {"confirmed": False}


def ports():
    return [{"id": identity, "roi_kind": "wire_entry_port", "bbox_xyxy": box, **REVIEW}
            for identity, box in [("A", [16, 26, 24, 34]), ("B", [96, 26, 104, 34]),
                                  ("C", [96, 76, 104, 84])]]


def mask(points, width=1):
    out = np.zeros((120, 120), np.uint8)
    cv2.polylines(out, [np.asarray(points, np.int32)], False, 255, width)
    return out


def view(masks, score=.9, entries=None, coverage=REVIEW):
    records = [extract_mask(m, score, "mask_" + str(i)) for i, m in enumerate(masks)]
    return assess_view(records, entries or ports(), [120, 120], coverage)


def compare(a, b, expectation=None):
    return compare_views(a, b, expectation if expectation is not None else [{"from": "A", "to": "B"}], REVIEW)


class Tests(unittest.TestCase):
    def setUp(self):
        self.line = mask([(20, 30), (100, 30)])
        self.reroute = mask([(20, 30), (20, 60), (100, 60), (100, 30)])
        self.wrong = mask([(20, 30), (20, 80), (100, 80)])

    def test_route_shape_does_not_change_endpoint_relation(self):
        r = compare(view([self.line]), view([self.reroute]))
        self.assertEqual(r["decision"], "same_visible_terminal_relations")
        self.assertTrue(r["visual_difference_may_be_route_only"])
        self.assertFalse(r["automatic_fault_verdict"])
        self.assertEqual(r["electrical_continuity"], "not_assessed")

    def test_changed_terminal(self):
        r = compare(view([self.line]), view([self.wrong]))
        self.assertEqual(r["decision"], "visible_terminal_relation_difference")
        self.assertEqual(r["missing_visible_relations"], [["A", "B"]])
        self.assertEqual(r["unexpected_visible_relations"], [["A", "C"]])

    def test_occlusion_is_not_guessed(self):
        broken = self.line.copy()
        broken[:, 50:70] = 0
        self.assertEqual(compare(view([self.line]), view([broken]))["decision"], "insufficient_evidence")

    def test_t_branch_not_pruned(self):
        branched = self.line.copy()
        branched[30:81, 60] = 255
        self.assertEqual(trace_skeleton(branched)["state"], "branched_or_crossing")
        self.assertFalse(view([branched])["complete_visible_scope"])

    def test_diagonal_crossing_rejects(self):
        crossed = mask([(20, 20), (90, 90)]) | mask([(90, 20), (20, 90)])
        self.assertEqual(trace_skeleton(crossed)["state"], "branched_or_crossing")

    def test_elbow_diagonal_shortcuts_are_removed(self):
        g = trace_skeleton(self.reroute)
        self.assertEqual(g["state"], "simple_visible_path")
        self.assertGreater(g["redundant_diagonal_edges_removed"], 0)
        self.assertEqual(len(g["path_xy"]), int((self.reroute > 0).sum()))

    def test_diagonal_line_preserved(self):
        self.assertEqual(trace_skeleton(mask([(20, 20), (90, 90)]))["state"], "simple_visible_path")

    def test_all_3x3_occupancies_preserve_component_count(self):
        for bits in itertools.product([0, 1], repeat=9):
            a = np.asarray(bits, np.uint8).reshape(3, 3)
            expected = cv2.connectedComponents(a, connectivity=8)[0] - 1
            self.assertEqual(trace_skeleton(a)["component_count"], expected)

    def test_rigid_grid_rotations_and_reflections(self):
        original = trace_skeleton(self.reroute)
        for k in range(4):
            for arr in (np.rot90(self.reroute, k), np.fliplr(np.rot90(self.reroute, k))):
                g = trace_skeleton(arr)
                self.assertEqual(g["state"], original["state"])
                self.assertEqual(g["skeleton_pixels"], original["skeleton_pixels"])
                self.assertEqual(g["redundant_diagonal_edges_removed"], original["redundant_diagonal_edges_removed"])

    def test_translation_and_changed_identity(self):
        entries = ports()
        for i, p in enumerate(entries):
            p["id"] = str(i)
            p["bbox_xyxy"] = [v + 5 for v in p["bbox_xyxy"]]
        moved = np.zeros_like(self.reroute)
        moved[5:, 5:] = self.reroute[:-5, :-5]
        r = view([moved], entries=entries)
        self.assertTrue(r["complete_visible_scope"])
        self.assertEqual((r["edges"][0]["from"], r["edges"][0]["to"]), ("0", "1"))

    def test_empty_not_normal_or_match(self):
        self.assertEqual(compare(view([]), view([]), expectation=[])["decision"], "insufficient_evidence")

    def test_loop_rejects(self):
        a = mask([(30, 30), (80, 30), (80, 80), (30, 80), (30, 30)])
        self.assertEqual(trace_skeleton(a)["state"], "not_two_ended")

    def test_frame_boundary_rejects(self):
        self.assertFalse(view([mask([(0, 30), (100, 30)])])["complete_visible_scope"])

    def test_overlapping_port_rois_reject(self):
        p = ports()
        p[2]["bbox_xyxy"] = p[1]["bbox_xyxy"][:]
        self.assertFalse(view([self.line], entries=p)["complete_visible_scope"])

    def test_unassigned_end_not_nearest_guessed(self):
        self.assertFalse(view([mask([(20, 30), (85, 30)])])["complete_visible_scope"])

    def test_body_box_not_port(self):
        p = ports()
        p[0]["roi_kind"] = "body_object"
        with self.assertRaises(ValueError):
            view([self.line], entries=p)

    def test_unconfirmed_ports_do_not_generate_edges(self):
        p = ports()
        p[0]["confirmed"] = False
        self.assertEqual(view([self.line], entries=p)["edges"], [])

    def test_coverage_requires_separate_confirmation(self):
        self.assertFalse(view([self.line], coverage=PENDING)["complete_visible_scope"])

    def test_expected_unknown_or_not_reviewed(self):
        v = view([self.line])
        self.assertEqual(compare_views(v, v, None, PENDING)["decision"], "insufficient_evidence")
        with self.assertRaises(ValueError):
            compare_views(v, v, None, REVIEW)

    def test_reference_wrong_not_used_as_authoritative_truth(self):
        self.assertEqual(compare(view([self.wrong]), view([self.wrong]))["decision"], "insufficient_evidence")

    def test_duplicate_masks_are_not_extra_connections(self):
        v = view([self.line, self.line.copy()])
        self.assertEqual(len(v["edges"]), 1)
        self.assertEqual(len(v["edges"][0]["evidence_ids"]), 2)

    def test_conflicting_masks_do_not_choose_best_score(self):
        self.assertIn("conflicting_relations_at_port", view([self.line, self.wrong])["reasons"])

    def test_third_port_in_middle_rejects(self):
        p = ports()
        p[2]["bbox_xyxy"] = [50, 26, 60, 34]
        self.assertFalse(view([self.line], entries=p)["complete_visible_scope"])

    def test_low_score_not_promoted(self):
        self.assertFalse(view([self.line], score=.749)["complete_visible_scope"])
        self.assertTrue(view([self.line], score=.75)["complete_visible_scope"])

    def test_invalid_input_rejected(self):
        for score in (float("nan"), float("inf"), True, -1, 2):
            with self.assertRaises(ValueError):
                extract_mask(self.line, score, "m")
        with self.assertRaises(ValueError):
            trace_skeleton(np.full((2, 2), np.nan))
        p = ports()
        p[0]["bbox_xyxy"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_ports(p, [120, 120])

    def test_inputs_not_modified(self):
        p = ports()
        before = deepcopy(p)
        a = self.reroute.copy()
        view([a], entries=p)
        self.assertEqual(p, before)
        np.testing.assert_array_equal(a, self.reroute)

    def test_separate_crossing_mask_relations_reject(self):
        p = ports()[:2]
        p += [{"id": identity, "roi_kind": "wire_entry_port", "bbox_xyxy": box, **REVIEW}
              for identity, box in [("D", [56, 16, 64, 24]), ("E", [56, 96, 64, 104])]]
        r = view([self.line, mask([(60, 20), (60, 100)])], entries=p)
        self.assertIn("overlapping_different_relation_paths", r["reasons"])

    def test_reversed_endpoint_direction_no_difference(self):
        a = view([self.line])
        b = deepcopy(a)
        b["edges"][0]["from"], b["edges"][0]["to"] = "B", "A"
        # API accepts declared undirected edges even if a caller reverses them.
        r = compare_views(a, b, [{"from": "B", "to": "A"}], REVIEW)
        self.assertEqual(r["decision"], "same_visible_terminal_relations")


if __name__ == "__main__":
    unittest.main(verbosity=2)
