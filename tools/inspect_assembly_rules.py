"""Align a production photo with the existing pipeline, then run new assembly rules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))

from assembly_engine.assembly_template import load_template  # noqa: E402
from assembly_engine.observation import observe_template  # noqa: E402
from assembly_engine.reference_builder import read_image, write_image  # noqa: E402
from assembly_engine.rule_engine import evaluate_rules  # noqa: E402


def _draw_evidence(reference, template, rule_report):  # type: ignore[no-untyped-def]
    overlay = reference.copy()
    object_results: dict[str, list[dict]] = {}
    for result in rule_report["rule_results"]:
        object_results.setdefault(result["object_id"], []).append(result)
    for item in template.objects:
        left, top, right, bottom = (round(value * scale) for value, scale in zip(item.expected_region, (reference.shape[1], reference.shape[0], reference.shape[1], reference.shape[0])))
        statuses = {result["status"] for result in object_results.get(item.id, [])}
        color = (0, 170, 0) if statuses == {"pass"} else ((0, 0, 255) if "suspected_ng" in statuses else (0, 180, 255))
        cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
        cv2.putText(overlay, item.id, (left, max(18, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "config" / "current_cabinet_assembly_template.json")
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "assembly_rule_inspections")
    parser.add_argument("--already-aligned", action="store_true", help="Treat --inspection as already aligned to the reference image.")
    args = parser.parse_args()
    template = load_template(args.template)
    reference = read_image(Path(template.reference_image))
    inspection = read_image(args.inspection)
    output = args.output / args.inspection.stem
    output.mkdir(parents=True, exist_ok=True)
    # A reference image used as its own inspection is a useful deterministic
    # smoke test.  Do not pass it through a resampling homography: interpolation
    # can make an identical image look position-uncertain even though no
    # production observation has occurred.
    same_as_reference = args.inspection.resolve() == Path(template.reference_image).resolve()
    if same_as_reference:
        aligned = reference.copy()
        alignment = {"method": "reference_self_check", "alignment_quality": {"reliable": True, "reason": "inspection_is_reference_image"}}
    elif args.already_aligned:
        aligned = inspection
        alignment = {"method": "caller_supplied_aligned_image", "alignment_quality": {"reliable": True, "reason": "not_recomputed"}}
    else:
        from assembly_auto_review_robust_v3 import automatic_homography  # noqa: PLC0415

        aligned, alignment = automatic_homography(reference, inspection)
    if aligned is None:
        report = {"template_id": template.template_id, "decision": "manual_review", "headline": "定位不可靠，未执行装配规则判断", "alignment": alignment}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return
    observation_report = observe_template(reference, aligned, template, reference_self_check=same_as_reference)
    rule_report = evaluate_rules(template, observation_report)
    report = {"alignment": alignment, "observations": observation_report, "rules": rule_report}
    write_image(output / "aligned.jpg", aligned)
    write_image(output / "rule_evidence.jpg", _draw_evidence(reference, template, rule_report))
    (output / "observation_report.json").write_text(json.dumps(observation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "rule_report.json").write_text(json.dumps(rule_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": rule_report["decision"], "summary": rule_report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
