from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

from wire_harness_segmentation_poc import (  # noqa: E402
    _canonical_class,
    build_report,
    masks_from_predictions,
    normalize_predictions,
    render_overlay,
    save_artifacts,
)


class WireHarnessSegmentationPocTests(unittest.TestCase):
    def test_local_cable_subclasses_are_merged_into_cable(self) -> None:
        self.assertEqual(_canonical_class("AJ20_D6_Cable"), "cable")
        self.assertEqual(_canonical_class("MHEV_N11_Cable"), "cable")
        self.assertEqual(_canonical_class("Etiquette Familly MHEV_NC11"), "etiquette familly mhev nc11")

    def test_normalizes_polygon_and_box_predictions_without_dropping_unknown_labels(self) -> None:
        raw = {
            "predictions": [
                {
                    "class": "Cable",
                    "confidence": 0.91,
                    "x": 30,
                    "y": 25,
                    "width": 20,
                    "height": 30,
                    "points": [{"x": 20, "y": 10}, {"x": 40, "y": 10}, {"x": 38, "y": 40}],
                },
                {"class": "custom_harness_label", "confidence": 0.4, "x": 70, "y": 40, "width": 10, "height": 8},
            ]
        }
        records, warnings = normalize_predictions(raw, (100, 80))
        self.assertEqual(warnings, [])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["class"], "cable")
        self.assertEqual(records[0]["geometry"], "polygon")
        self.assertEqual(records[0]["box_xyxy"], [20.0, 10.0, 40.0, 40.0])
        self.assertEqual(records[1]["source_class"], "custom_harness_label")
        self.assertEqual(records[1]["geometry"], "box_fallback")

    def test_masks_are_class_specific_and_union_only_target_classes(self) -> None:
        records = [
            {"class": "cable", "polygon": [[5, 5], [30, 5], [30, 20]], "box_xyxy": [5, 5, 30, 20]},
            {"class": "connector", "polygon": [], "box_xyxy": [40, 30, 55, 45]},
            {"class": "custom", "polygon": [], "box_xyxy": [0, 0, 90, 70]},
        ]
        masks = masks_from_predictions(records, (100, 80))
        self.assertGreater(int(np.count_nonzero(masks["cable"])), 0)
        self.assertGreater(int(np.count_nonzero(masks["connector"])), 0)
        self.assertEqual(int(np.count_nonzero(masks["clip"])), 0)
        self.assertEqual(np.array_equal(masks["all_target"], np.maximum(masks["cable"], masks["connector"])), True)
        self.assertLess(int(np.count_nonzero(masks["all_target"])), 90 * 70)

    def test_report_is_evidence_only_and_overlay_preserves_image_shape(self) -> None:
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        records = [{"class": "clip", "source_class": "Clip", "confidence": 0.8, "polygon": [], "box_xyxy": [10, 12, 25, 30], "geometry": "box_fallback"}]
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.png"
            fixture.write_bytes(b"fixture-image-bytes")
            report = build_report(fixture, image, records, [], confidence_threshold=0.25)
        self.assertEqual(report["decision"], "evidence_only_manual_review")
        self.assertEqual(report["summary"]["target_class_counts"]["clip"], 1)
        overlay = render_overlay(image, records)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(overlay)), 0)

    def test_artifacts_include_class_masks_overlay_and_report(self) -> None:
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        records = [{"class": "cable", "source_class": "Cable", "confidence": 0.8, "polygon": [[3, 3], [20, 3], [20, 12]], "box_xyxy": [3, 3, 20, 12], "geometry": "polygon"}]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            source.write_bytes(b"fixture-image-bytes")
            report = build_report(source, image, records, [], confidence_threshold=0.25)
            paths = save_artifacts(Path(directory) / "artifacts", image, report)
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            saved_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
        self.assertEqual(saved_report["artifacts"]["mask_cable"], paths["mask_cable"])
        self.assertEqual(saved_report["decision"], "evidence_only_manual_review")


if __name__ == "__main__":
    unittest.main()
