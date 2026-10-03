"""Auditable Agent workflow around existing visual-review evidence."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .topology import ConnectionGraph, compare_topologies


class WorkflowError(ValueError):
    """Raised for an invalid task transition or unsafe conclusion."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{field} must be a non-empty string")
    return value.strip()


def _visual_evidence(report: dict[str, Any], source: str | Path | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise WorkflowError("visual report must be an object")
    decision = str(report.get("decision", "unknown"))
    alignment = report.get("alignment_quality", {})
    alignment_reliable = bool(alignment.get("reliable", decision != "alignment_uncertain_manual_review"))
    regions = report.get("review_regions", [])
    if not isinstance(regions, list):
        raise WorkflowError("visual report review_regions must be a list")
    if not alignment_reliable or decision == "alignment_uncertain_manual_review":
        recommended_action = "targeted_recapture_or_manual_alignment_review"
    elif regions:
        recommended_action = "human_review_candidates"
    else:
        recommended_action = "human_confirm_no_actionable_difference"
    return {
        "source_report": str(Path(source).resolve()) if source is not None else None,
        "source_decision": decision,
        "alignment_reliable": alignment_reliable,
        "candidate_count": len(regions),
        "candidate_ids": [str(item.get("id", index)) for index, item in enumerate(regions, start=1) if isinstance(item, dict)],
        "recommended_action": recommended_action,
        "human_review_required": True,
        "automatic_fault_verdict": False,
    }


class InspectionTask:
    """Stateful task whose exported report keeps claims and provenance separate."""

    _REVIEW_OUTCOMES = {"confirmed_difference", "no_actionable_difference", "recapture_required"}

    def __init__(self, task_id: str, scene_type: str, reference: str, inspection: str) -> None:
        self._report: dict[str, Any] = {
            "schema_version": 1,
            "task_id": _text(task_id, "task_id"),
            "scene_type": _text(scene_type, "scene_type"),
            "inputs": {
                "reference": _text(reference, "reference"),
                "inspection": _text(inspection, "inspection"),
            },
            "state": "created",
            "created_at": _now(),
            "updated_at": _now(),
            "claim_boundary": {
                "visual_candidates_are_fault_verdicts": False,
                "topology_mismatch_requires_human_confirmation": True,
                "field_accuracy_claimed": False,
            },
            "tool_calls": [],
            "machine_evidence": [],
            "human_conclusions": [],
            "topology_assessments": [],
            "repair_guidance": [],
            "history": [],
        }
        self._event("task_created", "created")

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> "InspectionTask":
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise WorkflowError("task report schema_version must be 1")
        for field in ("task_id", "scene_type", "state"):
            _text(report.get(field), field)
        inputs = report.get("inputs")
        if not isinstance(inputs, dict):
            raise WorkflowError("task report inputs must be an object")
        _text(inputs.get("reference"), "inputs.reference")
        _text(inputs.get("inspection"), "inputs.inspection")
        for field in (
            "tool_calls",
            "machine_evidence",
            "human_conclusions",
            "topology_assessments",
            "repair_guidance",
            "history",
        ):
            if not isinstance(report.get(field), list):
                raise WorkflowError(f"task report {field} must be a list")
        if not isinstance(report.get("claim_boundary"), dict):
            raise WorkflowError("task report claim_boundary must be an object")
        task = cls.__new__(cls)
        task._report = deepcopy(report)
        return task

    @classmethod
    def load(cls, path: str | Path) -> "InspectionTask":
        source = Path(path)
        try:
            report = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkflowError(f"cannot load task report: {source}") from error
        return cls.from_report(report)

    @property
    def state(self) -> str:
        return str(self._report["state"])

    def _event(self, event: str, state: str, details: dict[str, Any] | None = None) -> None:
        occurred_at = _now()
        self._report["state"] = state
        self._report["updated_at"] = occurred_at
        record: dict[str, Any] = {"event": event, "state": state, "occurred_at": occurred_at}
        if details:
            record["details"] = deepcopy(details)
        self._report["history"].append(record)

    def record_visual_analysis(
        self,
        report: dict[str, Any],
        *,
        tool_name: str = "cabinet_dino_sam3_review",
        source_report: str | Path | None = None,
    ) -> dict[str, Any]:
        if self.state not in {"created", "reinspection_running"}:
            raise WorkflowError(f"cannot record visual analysis while task is {self.state}")
        evidence = _visual_evidence(report, source_report)
        phase = "reinspection" if self.state == "reinspection_running" else "initial_inspection"
        evidence["phase"] = phase
        evidence["recorded_at"] = _now()
        self._report["machine_evidence"].append(evidence)
        self._report["tool_calls"].append(
            {
                "tool": _text(tool_name, "tool_name"),
                "phase": phase,
                "status": "completed",
                "recorded_at": evidence["recorded_at"],
                "source_report": evidence["source_report"],
            }
        )
        next_state = "awaiting_human_review"
        self._event("visual_analysis_recorded", next_state, {"phase": phase, "recommended_action": evidence["recommended_action"]})
        return deepcopy(evidence)

    def record_human_review(
        self,
        *,
        reviewer: str,
        outcome: str,
        notes: str,
        confirmed_candidate_ids: list[str] | None = None,
    ) -> None:
        if self.state != "awaiting_human_review":
            raise WorkflowError(f"cannot record human review while task is {self.state}")
        if outcome not in self._REVIEW_OUTCOMES:
            raise WorkflowError(f"unsupported human-review outcome: {outcome}")
        conclusion = {
            "reviewer": _text(reviewer, "reviewer"),
            "outcome": outcome,
            "notes": _text(notes, "notes"),
            "confirmed_candidate_ids": list(confirmed_candidate_ids or []),
            "recorded_at": _now(),
        }
        self._report["human_conclusions"].append(conclusion)
        if outcome == "confirmed_difference":
            next_state = "difference_confirmed"
        elif outcome == "recapture_required":
            next_state = "recapture_required"
        else:
            next_state = "completed_no_actionable_difference"
        self._event("human_review_recorded", next_state, {"outcome": outcome})

    def record_local_evidence_review(self, bundle, review, files, case, *, plan=None, allow_test_records=False):
        """Attach live-verified local opinions without confirming differences or edges."""
        from .local_evidence_bridge import prepare_attachment
        supplement = prepare_attachment(self.to_report(), bundle, review, files, case,
                                        plan=plan, allow_test_records=allow_test_records)
        existing = self._report.get("local_evidence_reviews", [])
        if not isinstance(existing, list):
            raise WorkflowError("local evidence collection must be a list")
        if any(row.get("attachment_id") == supplement["attachment_id"] for row in existing):
            raise WorkflowError("same local evidence already attached to this phase")
        self._report.setdefault("local_evidence_reviews", []).append(deepcopy(supplement))
        self._report["tool_calls"].append({"tool": "local_evidence_live_import", "status": "completed",
                                          "phase": supplement["phase"], "recorded_at": _now()})
        self._event("local_evidence_review_attached", self.state,
                    {"attachment_id": supplement["attachment_id"], "case": case})
        return deepcopy(supplement)

    def assess_topology(
        self,
        expected: ConnectionGraph,
        observed: ConnectionGraph,
        *,
        minimum_confidence: float = 0.75,
    ) -> dict[str, Any]:
        if self.state not in {"awaiting_human_review", "difference_confirmed"}:
            raise WorkflowError(f"cannot assess topology while task is {self.state}")
        if expected.scene_type != self._report["scene_type"]:
            raise WorkflowError(
                f"topology scene_type {expected.scene_type} does not match task scene_type {self._report['scene_type']}"
            )
        assessment = compare_topologies(expected, observed, minimum_confidence=minimum_confidence)
        assessment["recorded_at"] = _now()
        self._report["topology_assessments"].append(assessment)
        self._report["tool_calls"].append(
            {
                "tool": "limited_connection_topology_compare",
                "phase": "topology_assessment",
                "status": "completed",
                "recorded_at": assessment["recorded_at"],
            }
        )
        self._event("topology_assessed", self.state, {"decision": assessment["decision"]})
        return deepcopy(assessment)

    def add_repair_guidance(self, instruction: str, *, evidence_ids: list[str]) -> None:
        if self.state != "difference_confirmed":
            raise WorkflowError("repair guidance requires a human-confirmed difference")
        if not evidence_ids or not all(isinstance(item, str) and item.strip() for item in evidence_ids):
            raise WorkflowError("repair guidance requires traceable evidence_ids")
        confirmed_ids = {
            evidence_id
            for conclusion in self._report["human_conclusions"]
            if conclusion.get("outcome") == "confirmed_difference"
            for evidence_id in conclusion.get("confirmed_candidate_ids", [])
        }
        unknown_ids = [item for item in evidence_ids if item.strip() not in confirmed_ids]
        if unknown_ids:
            raise WorkflowError(
                "repair guidance evidence_ids were not human-confirmed: " + ", ".join(unknown_ids)
            )
        guidance = {
            "instruction": _text(instruction, "instruction"),
            "evidence_ids": [item.strip() for item in evidence_ids],
            "automatic_instruction": False,
            "recorded_at": _now(),
        }
        self._report["repair_guidance"].append(guidance)
        self._event("repair_guidance_recorded", "repair_guidance_ready")

    def start_reinspection(self, inspection: str) -> None:
        if self.state not in {"repair_guidance_ready", "recapture_required"}:
            raise WorkflowError(f"cannot start reinspection while task is {self.state}")
        self._report["inputs"]["inspection"] = _text(inspection, "inspection")
        self._event("reinspection_started", "reinspection_running")

    def to_report(self) -> dict[str, Any]:
        return deepcopy(self._report)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_report(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target
