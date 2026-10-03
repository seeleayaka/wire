from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import assembly_auto_review_robust as robust  # noqa: E402
from assembly_auto_review_robust_v2 import grid_limited_ecc  # noqa: E402


class LocalEccStateTests(unittest.TestCase):
    def test_identical_textured_image_marks_every_corrected_tile(self) -> None:
        generator = np.random.default_rng(7)
        gray = generator.integers(0, 256, size=(240, 240), dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        locally_aligned, valid, diagnostic = grid_limited_ecc(image, image.copy())
        accepted_mask = robust.LAST_LOCAL_ACCEPTED_MASK

        self.assertEqual(locally_aligned.shape, image.shape)
        self.assertTrue(np.all(valid == 255))
        self.assertEqual(accepted_mask.shape, image.shape[:2])
        self.assertTrue(np.all(accepted_mask == 255))
        self.assertEqual(diagnostic["accepted_tiles"], diagnostic["tile_count"])
        self.assertEqual(diagnostic["accepted_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
