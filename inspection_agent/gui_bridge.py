"""Small persistence bridge used by the existing PyQt review application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .workflow import InspectionTask


def create_task_from_visual_report(
    task_path: str | Path,
    *,
    task_id: str,
    scene_type: str,
    reference: str,
    inspection: str,
    visual_report: dict[str, Any],
    source_report: str | Path,
) -> InspectionTask:
    task = InspectionTask(task_id, scene_type, reference, inspection)
    task.record_visual_analysis(visual_report, source_report=source_report)
    task.save(task_path)
    return task


def record_human_decision(
    task_path: str | Path,
    *,
    reviewer: str,
    outcome: str,
    notes: str,
    confirmed_candidate_ids: list[str],
) -> InspectionTask:
    task = InspectionTask.load(task_path)
    task.record_human_review(
        reviewer=reviewer,
        outcome=outcome,
        notes=notes,
        confirmed_candidate_ids=confirmed_candidate_ids,
    )
    task.save(task_path)
    return task


def record_guidance(
    task_path: str | Path,
    *,
    instruction: str,
    evidence_ids: list[str],
) -> InspectionTask:
    task = InspectionTask.load(task_path)
    task.add_repair_guidance(instruction, evidence_ids=evidence_ids)
    task.save(task_path)
    return task


def append_reinspection(
    task_path: str | Path,
    *,
    inspection: str,
    visual_report: dict[str, Any],
    source_report: str | Path,
) -> InspectionTask:
    task = InspectionTask.load(task_path)
    task.start_reinspection(inspection)
    task.record_visual_analysis(visual_report, source_report=source_report)
    task.save(task_path)
    return task
