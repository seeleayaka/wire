from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from assembly_engine.mendeley_intelliman_evidence import (
    MendeleyIntelliManEvidenceError,
    dataset_image_label_from_image_name,
    evaluate_mendeley_intelliman_evidence,
    load_mendeley_yolo_labels,
    render_evidence_overlay,
)


def write_image(path: Path, width: int = 100, height: int = 80) -> None:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    if not ok:
        raise AssertionError("test image encoding failed")
    encoded.tofile(str(path))


def topology(width: int = 100, height: int = 80) -> dict:
    return {
        "decision": "topology_diagnostic_only",
        "input": {"image_size_width_height": [width, height]},
        "topology": {
            "nodes": [
                {"node_id": 7, "position_row_col": [40, 50], "degree": 1},
                {"node_id": 8, "position_row_col": [10, 10], "degree": 2},
            ],
            "paths": [
                {"path_id": "path_a", "points_row_col": [[5, 5], [40, 50], [75, 95]]},
                {"path_id": "path_b", "points_row_col": [[1, 75], [8, 90]]},
            ],
            "intersections": [{"position_row_col": [42, 49], "path_ids": ["path_a"]}],
            "branch_points": [{"position_row_col": [5, 5], "path_ids": ["path_a"]}],
        },
    }


class MendeleyIntelliManEvidenceTests(unittest.TestCase):
    def test_yolo_boxes_are_converted_without_assigning_class_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "damaged_001.txt"
            labels.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            loaded = load_mendeley_yolo_labels(labels, (100, 80))
        self.assertEqual(loaded[0].class_id, 2)
        self.assertEqual(loaded[0].bbox_xyxy, (40.0, 24.0, 60.0, 56.0))

    def test_filename_prefix_is_the_dataset_image_label_not_a_box_class(self) -> None:
        self.assertEqual(dataset_image_label_from_image_name("disconnected_001.JPG"), "disconnected")
        self.assertIsNone(dataset_image_label_from_image_name("unknown_001.JPG"))

    def test_topology_evidence_is_local_to_the_source_label_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, labels, topology_path = root / "damaged_001.png", root / "damaged_001.txt", root / "topology.json"
            write_image(image)
            labels.write_text("2 0.5 0.5 0.2 0.4\n3 0.2 0.85 0.08 0.08\n", encoding="utf-8")
            topology_path.write_text(json.dumps(topology()), encoding="utf-8")
            report = evaluate_mendeley_intelliman_evidence(image, labels, topology_path)
        self.assertEqual(report["source"]["fault_family_from_filename"], "damaged")
        self.assertEqual(report["source"]["dataset_image_label"], "damaged")
        self.assertEqual(report["summary"]["source_label_count"], 2)
        self.assertEqual(report["summary"]["labels_with_topology_evidence"], 1)
        self.assertEqual(report["label_evidence"][0]["topology_endpoint_node_ids_in_or_near_box"], ["7"])
        self.assertEqual(report["label_evidence"][1]["evidence_state"], "no_topology_evidence_near_source_label_manual_evaluation")
        self.assertTrue(report["manual_evaluation_only"])

    def test_dimension_mismatch_is_rejected_before_geometry_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, labels, topology_path = root / "disconnected_001.png", root / "disconnected_001.txt", root / "topology.json"
            write_image(image)
            labels.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            topology_path.write_text(json.dumps(topology(width=99)), encoding="utf-8")
            with self.assertRaisesRegex(MendeleyIntelliManEvidenceError, "dimensions do not match"):
                evaluate_mendeley_intelliman_evidence(image, labels, topology_path)

    def test_evidence_overlay_is_saved_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, labels, topology_path = root / "damaged_001.png", root / "damaged_001.txt", root / "topology.json"
            overlay = root / "evidence.png"
            write_image(image)
            labels.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            topology_path.write_text(json.dumps(topology()), encoding="utf-8")
            report = evaluate_mendeley_intelliman_evidence(image, labels, topology_path)
            render_evidence_overlay(image, report, overlay)
            rendered = cv2.imdecode(np.fromfile(str(overlay), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape[:2], (80, 100))

    def test_unknown_or_invalid_yolo_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "bad.txt"
            labels.write_text("not_a_class 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            with self.assertRaisesRegex(MendeleyIntelliManEvidenceError, "non-numeric"):
                load_mendeley_yolo_labels(labels, (100, 80))


if __name__ == "__main__":
    unittest.main()
