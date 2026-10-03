from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from assembly_engine.assembly_template import AssemblyTemplate, TemplateObject, TemplateRule, load_template, save_template
from assembly_engine.observation import observe_template
from assembly_engine.reference_builder import build_reference_template
from assembly_engine.rule_engine import evaluate_rules


def template_with_one_wire() -> AssemblyTemplate:
    item = TemplateObject(
        id="wire_001",
        kind="wire_candidate",
        color={"name": "red", "bgr": [0.0, 0.0, 255.0], "tolerance": 42.0},
        center_norm=[0.5, 0.5],
        direction_deg=90.0,
        expected_region=[0.35, 0.15, 0.65, 0.85],
        presence_threshold=0.55,
        position_tolerance_px=15.0,
        direction_tolerance_deg=20.0,
        candidate_confidence=0.9,
        reference_metrics={"appearance_support": 0.8},
    )
    rules = [
        TemplateRule(id="wire_001_presence", type="exists", object_id=item.id),
        TemplateRule(id="wire_001_color", type="color", object_id=item.id),
        TemplateRule(id="wire_001_position", type="position", object_id=item.id),
    ]
    return AssemblyTemplate("test", "reference.png", [160, 120], "automatic_single_reference", [item], rules)


class AssemblyEngineTests(unittest.TestCase):
    def test_template_round_trip(self) -> None:
        template = template_with_one_wire()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.json"
            save_template(template, path)
            self.assertEqual(load_template(path).to_dict(), template.to_dict())

    def test_reference_builder_finds_synthetic_wire(self) -> None:
        image = np.zeros((180, 240, 3), dtype=np.uint8)
        cv2.line(image, (70, 25), (72, 155), (0, 0, 255), 8)
        template, _overlay, report = build_reference_template(image, "synthetic.png", "synthetic")
        self.assertGreaterEqual(report["candidate_count"], 1)
        self.assertTrue(any(item.kind == "wire_candidate" for item in template.objects))

    def test_observation_and_rules_pass_for_matching_wire(self) -> None:
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.line(reference, (80, 20), (80, 102), (0, 0, 255), 8)
        report = observe_template(reference, reference.copy(), template_with_one_wire())
        verdict = evaluate_rules(template_with_one_wire(), report)
        self.assertEqual(verdict["decision"], "ok")

    def test_missing_wire_becomes_suspected_ng_when_difference_supports_it(self) -> None:
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.line(reference, (80, 20), (80, 102), (0, 0, 255), 8)
        inspection = np.zeros_like(reference)
        verdict = evaluate_rules(template_with_one_wire(), observe_template(reference, inspection, template_with_one_wire()))
        self.assertEqual(verdict["decision"], "suspected_ng")
        self.assertTrue(any(result["reason"].startswith("suspected_missing") for result in verdict["rule_results"]))

    def test_wrong_color_becomes_suspected_ng(self) -> None:
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        inspection = np.zeros_like(reference)
        cv2.line(reference, (80, 20), (80, 102), (0, 0, 255), 8)
        cv2.line(inspection, (80, 20), (80, 102), (255, 0, 0), 8)
        verdict = evaluate_rules(template_with_one_wire(), observe_template(reference, inspection, template_with_one_wire()))
        self.assertEqual(verdict["decision"], "suspected_ng")
        self.assertTrue(any(result["type"] == "color" and result["status"] == "suspected_ng" for result in verdict["rule_results"]))

    def test_position_deviation_requires_visual_support(self) -> None:
        base = template_with_one_wire()
        item = replace(base.objects[0], expected_region=[0.25, 0.15, 0.75, 0.85])
        template = AssemblyTemplate(base.template_id, base.reference_image, base.image_size, base.build_mode, [item], base.rules)
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        inspection = np.zeros_like(reference)
        cv2.line(reference, (80, 20), (80, 102), (0, 0, 255), 8)
        cv2.line(inspection, (110, 20), (110, 102), (0, 0, 255), 8)
        verdict = evaluate_rules(template, observe_template(reference, inspection, template))
        position_results = [result for result in verdict["rule_results"] if result["type"] == "position"]
        self.assertEqual(len(position_results), 1)
        self.assertIn(position_results[0]["status"], {"manual_review", "suspected_ng"})


if __name__ == "__main__":
    unittest.main()
