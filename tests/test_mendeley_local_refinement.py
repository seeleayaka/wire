from __future__ import annotations

import unittest

import cv2
import numpy as np

from tools.evaluate_mendeley_local_refinement import refine_one
from tools.probe_mendeley_local_heat import decode_jet


class MendeleyLocalRefinementTests(unittest.TestCase):
    def test_jet_decode_round_trip(self) -> None:
        values = np.arange(256, dtype=np.uint8).reshape(16, 16)
        heat = cv2.applyColorMap(values, cv2.COLORMAP_JET)
        np.testing.assert_array_equal(decode_jet(heat), values)

    def test_refine_one_selects_strong_local_component(self) -> None:
        score = np.zeros((100, 100), dtype=np.uint8)
        score[30:40, 45:55] = 250
        candidate = {"left": 10, "top": 10, "right": 90, "bottom": 90}
        refined = refine_one(candidate, score, percentile=90, minimum=200, min_pixels=20, margin=0)
        self.assertEqual([refined[key] for key in ("left", "top", "right", "bottom")], [45, 30, 55, 40])

    def test_refine_one_preserves_parent_when_no_component(self) -> None:
        score = np.zeros((100, 100), dtype=np.uint8)
        candidate = {"left": 10, "top": 10, "right": 90, "bottom": 90}
        self.assertEqual(refine_one(candidate, score, percentile=90, minimum=200, min_pixels=20, margin=0), candidate)


if __name__ == "__main__":
    unittest.main()
