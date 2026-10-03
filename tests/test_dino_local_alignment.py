from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import assembly_auto_review_dino as dino  # noqa: E402


class DinoLocalAlignmentTests(unittest.TestCase):
    def _run_with_local_state(self, coverage: float, *, accepted: bool) -> tuple[np.ndarray, list[dict]]:
        reference = np.zeros((12, 16, 3), dtype=np.uint8)
        globally_aligned = np.full_like(reference, 20)
        locally_aligned = np.full_like(reference, 180)
        accepted_mask = np.zeros(reference.shape[:2], dtype=np.uint8)
        accepted_mask[:, :8] = 255
        captured: list[np.ndarray] = []

        def prepare_local_alignment(*_args):
            dino.robust.LAST_DIAGNOSTICS = [{"accepted_coverage": coverage, "accepted": accepted}]
            dino.robust.LAST_LOCAL_ALIGNED = locally_aligned
            dino.robust.LAST_LOCAL_ACCEPTED_MASK = accepted_mask
            dino.robust.LAST_LOCAL_ALIGNMENT_PARTS = []
            return globally_aligned.copy(), np.zeros_like(reference), []

        def capture_review(_reference_part, aligned_part, *_args, **_kwargs):
            captured.append(aligned_part.copy())
            return np.zeros(reference.shape[:2], dtype=np.float32), {}, []

        with (
            patch.object(dino.adaptive, "adaptive_regions", side_effect=prepare_local_alignment),
            patch.object(dino, "review_components", side_effect=capture_review),
        ):
            dino.dino_fused_regions(reference, globally_aligned, [[0.0, 0.0, 1.0, 1.0]])

        return captured[0], dino.robust.LAST_DIAGNOSTICS

    def test_reliable_local_alignment_replaces_only_accepted_pixels(self) -> None:
        dino_input, diagnostics = self._run_with_local_state(coverage=0.67, accepted=True)

        self.assertTrue(np.all(dino_input[:, :8] == 180))
        self.assertTrue(np.all(dino_input[:, 8:] == 20))
        self.assertEqual(diagnostics[0]["dino_alignment_input"], "local_ecc_corrected")

    def test_low_local_coverage_keeps_global_alignment(self) -> None:
        dino_input, diagnostics = self._run_with_local_state(coverage=0.50, accepted=False)

        self.assertTrue(np.all(dino_input == 20))
        self.assertEqual(diagnostics[0]["dino_alignment_input"], "global_homography")

    def test_roi_specific_local_state_wins_over_full_image_compatibility_state(self) -> None:
        reference = np.zeros((12, 16, 3), dtype=np.uint8)
        globally_aligned = np.full_like(reference, 20)
        full_compatibility_image = np.full_like(reference, 180)
        full_compatibility_mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
        roi_aligned = np.full_like(reference, 220)
        roi_mask = np.zeros(reference.shape[:2], dtype=np.uint8)
        roi_mask[:, 8:] = 255
        captured: list[np.ndarray] = []

        def prepare_local_alignment(*_args):
            dino.robust.LAST_DIAGNOSTICS = [{"accepted_coverage": 0.67, "accepted": True}]
            dino.robust.LAST_LOCAL_ALIGNED = full_compatibility_image
            dino.robust.LAST_LOCAL_ACCEPTED_MASK = full_compatibility_mask
            dino.robust.LAST_LOCAL_ALIGNMENT_PARTS = [{
                "bounds": [0, 0, 16, 12],
                "aligned": roi_aligned,
                "accepted_mask": roi_mask,
            }]
            return globally_aligned.copy(), np.zeros_like(reference), []

        def capture_review(_reference_part, aligned_part, *_args, **_kwargs):
            captured.append(aligned_part.copy())
            return np.zeros(reference.shape[:2], dtype=np.float32), {}, []

        with (
            patch.object(dino.adaptive, "adaptive_regions", side_effect=prepare_local_alignment),
            patch.object(dino, "review_components", side_effect=capture_review),
        ):
            dino.dino_fused_regions(reference, globally_aligned, [[0.0, 0.0, 1.0, 1.0]])

        self.assertTrue(np.all(captured[0][:, :8] == 20))
        self.assertTrue(np.all(captured[0][:, 8:] == 220))


if __name__ == "__main__":
    unittest.main()
