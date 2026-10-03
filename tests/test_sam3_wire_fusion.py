from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import sam3_wire_fusion as fusion  # noqa: E402


class Sam3WireFusionTests(unittest.TestCase):
    def test_platform_specific_python_path_selects_the_target_runtime(self) -> None:
        configured = {
            "windows": "runtime/sam3/.venv/Scripts/python.exe",
            "linux": "runtime/sam3/.venv/bin/python",
        }
        self.assertEqual(
            fusion._path_for_platform(configured, "sam3_python", "windows"),
            ROOT / "runtime/sam3/.venv/Scripts/python.exe",
        )
        self.assertEqual(
            fusion._path_for_platform(configured, "sam3_python", "linux"),
            ROOT / "runtime/sam3/.venv/bin/python",
        )

    def test_recipe_points_to_the_isolated_verified_resources(self) -> None:
        settings = fusion.load_settings()
        self.assertTrue(settings.python.is_file())
        self.assertTrue((settings.source_root / "sam3").is_dir())
        self.assertTrue(settings.runner.is_file())
        self.assertTrue(settings.checkpoint.is_file())
        self.assertEqual(settings.confidence_threshold, 0.4)

    def test_component_records_keep_disconnected_masks_separate(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:7, 2:6] = True
        mask[12:18, 12:18] = True
        records = fusion._component_records(mask, "inspection_only", minimum_pixels=10, minimum_span=4)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0]["direction"], "inspection_only")
        self.assertEqual(records[0][0]["bbox_xyxy"], [2, 2, 6, 7])
        self.assertEqual(records[1][0]["bbox_xyxy"], [12, 12, 18, 18])

    def test_jet_score_round_trip_uses_the_dino_heat_encoding(self) -> None:
        values = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
        heat = cv2.applyColorMap(values, cv2.COLORMAP_JET)
        recovered = fusion._jet_to_score(heat)
        self.assertTrue(np.array_equal(recovered, values))

    def test_overlap_is_strict_for_touching_boxes(self) -> None:
        self.assertTrue(fusion._overlap((1, 1, 5, 5), (4, 4, 8, 8)))
        self.assertFalse(fusion._overlap((1, 1, 5, 5), (5, 1, 8, 5)))

    def test_sam_difference_render_uses_only_sam_components(self) -> None:
        aligned = np.zeros((20, 20, 3), dtype=np.uint8)
        added = np.zeros((20, 20), dtype=bool)
        missing = np.zeros((20, 20), dtype=bool)
        added[2:8, 2:8] = True
        missing[12:18, 12:18] = True
        components = [
            {"direction": "inspection_only", "bbox_xyxy": [2, 2, 8, 8]},
            {"direction": "reference_only", "bbox_xyxy": [12, 12, 18, 18]},
        ]

        rendered = fusion._render_sam_difference(aligned, added, missing, components)

        self.assertEqual(rendered.shape, aligned.shape)
        self.assertGreater(int(rendered.sum()), 0)

    def test_dino_candidate_without_strong_sam_support_is_retained_for_manual_review(self) -> None:
        added = np.zeros((12, 12), dtype=bool)
        missing = np.zeros((12, 12), dtype=bool)
        added[7:11, 7:11] = True
        candidates = [
            {"left": 1, "top": 1, "right": 5, "bottom": 5, "evidence_scores": {"dino": 91.0}},
            {"left": 7, "top": 7, "right": 11, "bottom": 11, "evidence_scores": {"dino": 88.0}},
        ]

        green, orange = fusion._tier_dino_candidates(
            candidates, added, missing, minimum_sam_support_pixels=8
        )

        self.assertEqual([item["id"] for item in green], ["green_01"])
        self.assertEqual(green[0]["source_dino_candidate_index"], 2)
        self.assertEqual([item["id"] for item in orange], ["orange_01"])
        self.assertEqual(orange[0]["source_dino_candidate_index"], 1)
        self.assertEqual(orange[0]["sam_support_pixels"], 0)
        self.assertEqual(orange[0]["tier"], "dino_only_human_review")


if __name__ == "__main__":
    unittest.main()
