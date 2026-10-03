from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from inspection_agent import ConnectionGraph, InspectionTask, TopologyValidationError, WorkflowError, compare_topologies
from inspection_agent import gui_bridge


def graph(graph_id: str, connections: list[dict], scene_type: str = "bench_terminal_board") -> ConnectionGraph:
    return ConnectionGraph.from_dict(
        {
            "schema_version": 1,
            "graph_id": graph_id,
            "scene_type": scene_type,
            "nodes": [
                {"id": "power_plus", "kind": "terminal"},
                {"id": "load_in", "kind": "terminal"},
                {"id": "load_out", "kind": "terminal"},
                {"id": "power_minus", "kind": "terminal"},
            ],
            "connections": connections,
        }
    )


class ConnectionGraphTests(unittest.TestCase):
    def test_match_is_order_independent(self) -> None:
        expected = graph("expected", [{"id": "e1", "from": "power_plus", "to": "load_in"}])
        observed = graph(
            "observed",
            [{"id": "o1", "from": "load_in", "to": "power_plus", "confidence": 0.96, "evidence_ids": ["segment_01"]}],
        )
        result = compare_topologies(expected, observed)
        self.assertEqual(result["decision"], "match")
        self.assertFalse(result["automatic_fault_verdict"])
        self.assertEqual(result["topology_distance"]["normalized_distance"], 0.0)
        self.assertEqual(result["topology_distance"]["similarity"], 1.0)

    def test_missing_and_unexpected_connections_are_explainable(self) -> None:
        expected = graph("expected", [{"id": "e1", "from": "power_plus", "to": "load_in"}])
        observed = graph("observed", [{"id": "o1", "from": "power_plus", "to": "load_out", "confidence": 0.9}])
        result = compare_topologies(expected, observed)
        self.assertEqual(result["decision"], "mismatch_manual_confirmation_required")
        self.assertEqual(result["summary"]["missing_count"], 1)
        self.assertEqual(result["summary"]["unexpected_count"], 1)
        self.assertEqual(result["topology_distance"]["edge_edit_count"], 2)
        self.assertEqual(result["topology_distance"]["normalized_distance"], 1.0)

    def test_normalized_edge_distance_reports_partial_overlap(self) -> None:
        expected = graph(
            "expected",
            [
                {"id": "e1", "from": "power_plus", "to": "load_in"},
                {"id": "e2", "from": "load_out", "to": "power_minus"},
            ],
        )
        observed = graph(
            "observed",
            [{"id": "o1", "from": "power_plus", "to": "load_in", "confidence": 0.95}],
        )
        result = compare_topologies(expected, observed)
        distance = result["topology_distance"]
        self.assertEqual(distance["metric"], "normalized_symmetric_edge_difference")
        self.assertFalse(distance["is_image2net_ned"])
        self.assertEqual(distance["edge_edit_count"], 1)
        self.assertEqual(distance["edge_union_count"], 2)
        self.assertEqual(distance["normalized_distance"], 0.5)
        self.assertEqual(distance["similarity"], 0.5)

    def test_low_confidence_connection_preserves_uncertainty(self) -> None:
        expected = graph("expected", [{"id": "e1", "from": "power_plus", "to": "load_in"}])
        observed = graph("observed", [{"id": "o1", "from": "power_plus", "to": "load_in", "confidence": 0.4}])
        result = compare_topologies(expected, observed)
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(result["summary"]["uncertain_count"], 1)
        self.assertEqual(result["topology_distance"]["uncertain_connections_excluded"], 1)

    def test_undeclared_node_is_rejected(self) -> None:
        with self.assertRaises(TopologyValidationError):
            graph("bad", [{"id": "e1", "from": "power_plus", "to": "unknown"}])


class InspectionTaskTests(unittest.TestCase):
    def test_visual_candidate_remains_machine_evidence_until_human_review(self) -> None:
        task = InspectionTask("task-001", "cabinet_harness", "right.png", "wrong.png")
        evidence = task.record_visual_analysis(
            {
                "decision": "possible_difference_manual_review",
                "alignment_quality": {"reliable": True},
                "review_regions": [{"id": "green_01"}],
            },
            source_report="output/report.json",
        )
        self.assertTrue(evidence["human_review_required"])
        self.assertFalse(evidence["automatic_fault_verdict"])
        self.assertEqual(task.state, "awaiting_human_review")

        task.record_human_review(
            reviewer="operator-01",
            outcome="confirmed_difference",
            notes="可见游离线已人工确认",
            confirmed_candidate_ids=["green_01"],
        )
        self.assertEqual(task.state, "difference_confirmed")

    def test_alignment_failure_routes_to_recapture(self) -> None:
        task = InspectionTask("task-002", "cabinet_harness", "right.png", "angled.png")
        evidence = task.record_visual_analysis(
            {
                "decision": "alignment_uncertain_manual_review",
                "alignment_quality": {"reliable": False},
                "review_regions": [],
            }
        )
        self.assertEqual(evidence["recommended_action"], "targeted_recapture_or_manual_alignment_review")
        task.record_human_review(
            reviewer="operator-01",
            outcome="recapture_required",
            notes="请保持正视角重新拍摄上半区",
        )
        self.assertEqual(task.state, "recapture_required")
        task.start_reinspection("recaptured.png")
        self.assertEqual(task.state, "reinspection_running")

    def test_repair_guidance_requires_human_confirmation_and_evidence(self) -> None:
        task = InspectionTask("task-003", "bench_terminal_board", "expected.png", "observed.png")
        with self.assertRaises(WorkflowError):
            task.add_repair_guidance("移动导线", evidence_ids=["edge_01"])
        task.record_visual_analysis({"decision": "possible_difference_manual_review", "review_regions": [{"id": "edge_01"}]})
        task.record_human_review(
            reviewer="operator-01",
            outcome="confirmed_difference",
            notes="导线插在错误端子",
            confirmed_candidate_ids=["edge_01"],
        )
        with self.assertRaises(WorkflowError):
            task.add_repair_guidance("移动导线", evidence_ids=[])
        with self.assertRaises(WorkflowError):
            task.add_repair_guidance("移动导线", evidence_ids=["unconfirmed_99"])
        task.add_repair_guidance("将导线从 load_out 移至 load_in", evidence_ids=["edge_01"])
        report = task.to_report()
        self.assertEqual(report["state"], "repair_guidance_ready")
        self.assertFalse(report["repair_guidance"][0]["automatic_instruction"])
        self.assertFalse(report["claim_boundary"]["field_accuracy_claimed"])

    def test_topology_assessment_is_logged_as_tool_call(self) -> None:
        task = InspectionTask("task-004", "bench_terminal_board", "expected.png", "observed.png")
        task.record_visual_analysis({"decision": "possible_difference_manual_review", "review_regions": [{"id": "segment_01"}]})
        expected = graph("expected", [{"id": "e1", "from": "power_plus", "to": "load_in"}])
        observed = graph("observed", [{"id": "o1", "from": "power_plus", "to": "load_out", "confidence": 0.9}])
        result = task.assess_topology(expected, observed)
        self.assertEqual(result["decision"], "mismatch_manual_confirmation_required")
        self.assertEqual(task.to_report()["tool_calls"][-1]["tool"], "limited_connection_topology_compare")

    def test_topology_scene_must_match_task_scene(self) -> None:
        task = InspectionTask("task-scene", "cabinet_harness", "expected.png", "observed.png")
        task.record_visual_analysis({"decision": "possible_difference_manual_review", "review_regions": []})
        expected = graph("expected", [{"id": "e1", "from": "power_plus", "to": "load_in"}])
        observed = graph("observed", [{"id": "o1", "from": "power_plus", "to": "load_in"}])
        with self.assertRaises(WorkflowError):
            task.assess_topology(expected, observed)

    def test_task_report_round_trip_preserves_state_and_chinese(self) -> None:
        task = InspectionTask("task-005", "cabinet_harness", "基准图.png", "待检图.png")
        task.record_visual_analysis({"decision": "no_significant_wire_related_difference", "review_regions": []})
        task.record_human_review(
            reviewer="操作员甲",
            outcome="no_actionable_difference",
            notes="人工确认没有需要处理的变化",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            task.save(path)
            restored = InspectionTask.load(path)
        self.assertEqual(restored.state, "completed_no_actionable_difference")
        self.assertEqual(restored.to_report()["human_conclusions"][0]["reviewer"], "操作员甲")

    def test_gui_bridge_appends_reinspection_to_same_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory) / "agent_task.json"
            task = gui_bridge.create_task_from_visual_report(
                task_path,
                task_id="task-reinspect",
                scene_type="cabinet_harness",
                reference="right.png",
                inspection="wrong.png",
                visual_report={"decision": "possible_difference_manual_review", "review_regions": [{"id": "green_01"}]},
                source_report=Path(directory) / "first_report.json",
            )
            task.record_human_review(
                reviewer="operator-01",
                outcome="confirmed_difference",
                notes="确认存在可见差异",
                confirmed_candidate_ids=["green_01"],
            )
            task.add_repair_guidance("重新接线后复拍", evidence_ids=["green_01"])
            task.save(task_path)
            updated = gui_bridge.append_reinspection(
                task_path,
                inspection="corrected.png",
                visual_report={"decision": "no_significant_wire_related_difference", "review_regions": []},
                source_report=Path(directory) / "second_report.json",
            )
        report = updated.to_report()
        self.assertEqual(updated.state, "awaiting_human_review")
        self.assertEqual(len(report["machine_evidence"]), 2)
        self.assertEqual(report["machine_evidence"][-1]["phase"], "reinspection")


if __name__ == "__main__":
    unittest.main()
