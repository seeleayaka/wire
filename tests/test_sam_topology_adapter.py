from __future__ import annotations

import unittest

from inspection_agent import (
    SamTopologyAdapterError,
    compare_topologies,
    connection_graph_from_sam_endpoints,
)


def terminal_map(*terminals: tuple[str, list[int]]) -> dict:
    return {
        "schema_version": 1,
        "graph_id": "controlled-observed",
        "scene_type": "bench_terminal_board",
        "terminals": [
            {"id": terminal_id, "label": terminal_id, "bbox_xyxy": bbox}
            for terminal_id, bbox in terminals
        ],
    }


def endpoint_report(*records: tuple[str, float, list[list[int]]]) -> dict:
    return {
        "source_dir": "synthetic_sam3_contract_fixture",
        "records": [
            {
                "record_id": record_id,
                "source_score": score,
                "visible_ends_xy": ends,
                "endpoint_status": "unclassified_visible_segment_end",
            }
            for record_id, score, ends in records
        ],
    }


class SamTopologyAdapterTests(unittest.TestCase):
    def test_unique_endpoint_assignments_create_matching_edge(self) -> None:
        terminals = terminal_map(("T1", [0, 0, 20, 20]), ("T2", [80, 0, 100, 20]))
        source = endpoint_report(("m001_c01", 0.93, [[10, 10], [90, 10]]))

        observed, audit = connection_graph_from_sam_endpoints(source, terminals)
        expected = type(observed).from_dict(
            {
                "schema_version": 1,
                "graph_id": "expected",
                "scene_type": "bench_terminal_board",
                "nodes": [{"id": "T1"}, {"id": "T2"}],
                "connections": [{"id": "expected_1", "from": "T1", "to": "T2"}],
            }
        )

        self.assertEqual(compare_topologies(expected, observed)["decision"], "match")
        self.assertEqual(observed.connections[0].confidence, 0.93)
        self.assertEqual(observed.connections[0].evidence_ids, ("m001_c01",))
        self.assertEqual(audit["evidence_records"][0]["state"], "candidate_connection")
        self.assertFalse(audit["automatic_fault_verdict"])

    def test_unassigned_endpoint_does_not_create_edge(self) -> None:
        terminals = terminal_map(("T1", [0, 0, 20, 20]), ("T2", [80, 0, 100, 20]))
        source = endpoint_report(("m001_c01", 0.93, [[10, 10], [50, 50]]))
        observed, audit = connection_graph_from_sam_endpoints(source, terminals)
        self.assertEqual(observed.connections, ())
        self.assertEqual(audit["evidence_records"][0]["state"], "endpoint_unassigned")

    def test_overlapping_terminal_rois_remain_ambiguous(self) -> None:
        terminals = terminal_map(
            ("T1", [0, 0, 20, 20]),
            ("T2", [5, 5, 25, 25]),
            ("T3", [80, 0, 100, 20]),
        )
        source = endpoint_report(("m001_c01", 0.93, [[10, 10], [90, 10]]))
        observed, audit = connection_graph_from_sam_endpoints(source, terminals)
        self.assertEqual(observed.connections, ())
        self.assertEqual(
            audit["evidence_records"][0]["state"],
            "endpoint_matches_multiple_terminals",
        )

    def test_same_terminal_at_both_ends_does_not_create_self_edge(self) -> None:
        terminals = terminal_map(("T1", [0, 0, 40, 40]))
        source = endpoint_report(("m001_c01", 0.93, [[10, 10], [30, 30]]))
        observed, audit = connection_graph_from_sam_endpoints(source, terminals)
        self.assertEqual(observed.connections, ())
        self.assertEqual(audit["evidence_records"][0]["state"], "same_terminal_both_ends")

    def test_duplicate_segment_evidence_is_merged_into_one_edge(self) -> None:
        terminals = terminal_map(("T1", [0, 0, 20, 20]), ("T2", [80, 0, 100, 20]))
        source = endpoint_report(
            ("m001_c01", 0.81, [[10, 10], [90, 10]]),
            ("m002_c01", 0.94, [[11, 11], [89, 11]]),
        )
        observed, audit = connection_graph_from_sam_endpoints(source, terminals)
        self.assertEqual(len(observed.connections), 1)
        self.assertEqual(observed.connections[0].confidence, 0.94)
        self.assertEqual(observed.connections[0].evidence_ids, ("m001_c01", "m002_c01"))
        self.assertEqual(audit["summary"]["merged_duplicate_evidence_count"], 1)

    def test_low_sam_score_is_preserved_as_insufficient_evidence(self) -> None:
        terminals = terminal_map(("T1", [0, 0, 20, 20]), ("T2", [80, 0, 100, 20]))
        source = endpoint_report(("m001_c01", 0.53, [[10, 10], [90, 10]]))
        observed, _ = connection_graph_from_sam_endpoints(source, terminals)
        expected = type(observed).from_dict(
            {
                "schema_version": 1,
                "graph_id": "expected",
                "scene_type": "bench_terminal_board",
                "nodes": [{"id": "T1"}, {"id": "T2"}],
                "connections": [{"id": "expected_1", "from": "T1", "to": "T2"}],
            }
        )
        comparison = compare_topologies(expected, observed, minimum_confidence=0.75)
        self.assertEqual(comparison["decision"], "insufficient_evidence")
        self.assertEqual(comparison["summary"]["uncertain_count"], 1)

    def test_invalid_terminal_box_and_padding_are_rejected(self) -> None:
        bad = terminal_map(("T1", [10, 10, 5, 20]))
        with self.assertRaises(SamTopologyAdapterError):
            connection_graph_from_sam_endpoints(endpoint_report(), bad)
        with self.assertRaises(SamTopologyAdapterError):
            connection_graph_from_sam_endpoints(
                endpoint_report(), terminal_map(("T1", [0, 0, 20, 20])), endpoint_padding_px=-1
            )


if __name__ == "__main__":
    unittest.main()
