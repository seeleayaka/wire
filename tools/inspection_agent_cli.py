"""Command-line task workflow for the Zhijie inspection Agent.

This entry point wraps existing visual reports. It never upgrades a machine
candidate to a confirmed fault without an explicit human-review command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection_agent import ConnectionGraph, InspectionTask  # noqa: E402


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _output_path(args: argparse.Namespace) -> Path:
    return args.output if args.output is not None else args.task


def _start(args: argparse.Namespace) -> None:
    task = InspectionTask(args.task_id, args.scene_type, args.reference, args.inspection)
    task.record_visual_analysis(
        _json_object(args.visual_report),
        source_report=args.visual_report,
    )
    task.save(args.output)


def _review(args: argparse.Namespace) -> None:
    task = InspectionTask.load(args.task)
    task.record_human_review(
        reviewer=args.reviewer,
        outcome=args.outcome,
        notes=args.notes,
        confirmed_candidate_ids=args.candidate,
    )
    task.save(_output_path(args))


def _topology(args: argparse.Namespace) -> None:
    task = InspectionTask.load(args.task)
    expected = ConnectionGraph.from_dict(_json_object(args.expected))
    observed = ConnectionGraph.from_dict(_json_object(args.observed))
    task.assess_topology(expected, observed, minimum_confidence=args.minimum_confidence)
    task.save(_output_path(args))


def _guidance(args: argparse.Namespace) -> None:
    task = InspectionTask.load(args.task)
    task.add_repair_guidance(args.instruction, evidence_ids=args.evidence_id)
    task.save(_output_path(args))


def _reinspect(args: argparse.Namespace) -> None:
    task = InspectionTask.load(args.task)
    task.start_reinspection(args.inspection)
    task.record_visual_analysis(
        _json_object(args.visual_report),
        source_report=args.visual_report,
    )
    task.save(_output_path(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="智接 inspection Agent task workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="create a task from an existing visual report")
    start.add_argument("--task-id", required=True)
    start.add_argument("--scene-type", choices=("cabinet_harness", "bench_terminal_board"), required=True)
    start.add_argument("--reference", required=True)
    start.add_argument("--inspection", required=True)
    start.add_argument("--visual-report", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    start.set_defaults(handler=_start)

    review = subparsers.add_parser("review", help="record an explicit human conclusion")
    review.add_argument("--task", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument(
        "--outcome",
        choices=("confirmed_difference", "no_actionable_difference", "recapture_required"),
        required=True,
    )
    review.add_argument("--notes", required=True)
    review.add_argument("--candidate", action="append", default=[])
    review.add_argument("--output", type=Path)
    review.set_defaults(handler=_review)

    topology = subparsers.add_parser("topology", help="compare declared expected and observed connections")
    topology.add_argument("--task", type=Path, required=True)
    topology.add_argument("--expected", type=Path, required=True)
    topology.add_argument("--observed", type=Path, required=True)
    topology.add_argument("--minimum-confidence", type=float, default=0.75)
    topology.add_argument("--output", type=Path)
    topology.set_defaults(handler=_topology)

    guidance = subparsers.add_parser("guidance", help="record guidance after a human-confirmed difference")
    guidance.add_argument("--task", type=Path, required=True)
    guidance.add_argument("--instruction", required=True)
    guidance.add_argument("--evidence-id", action="append", required=True)
    guidance.add_argument("--output", type=Path)
    guidance.set_defaults(handler=_guidance)

    reinspect = subparsers.add_parser("reinspect", help="attach a new visual report after correction or recapture")
    reinspect.add_argument("--task", type=Path, required=True)
    reinspect.add_argument("--inspection", required=True)
    reinspect.add_argument("--visual-report", type=Path, required=True)
    reinspect.add_argument("--output", type=Path)
    reinspect.set_defaults(handler=_reinspect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.handler(args)
    target = args.output if args.output is not None else getattr(args, "task", None)
    print(json.dumps({"status": "ok", "command": args.command, "task_report": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
