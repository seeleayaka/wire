from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import dino_feature_diff as feature_diff


class DinoLowAbsoluteSignalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.full((40, 60, 3), 120, dtype=np.uint8)

    def test_near_zero_photometric_evidence_is_not_percentile_amplified(self) -> None:
        with patch.object(
            feature_diff,
            "dino_difference",
            return_value=(np.full((40, 60), 0.002, dtype=np.float32), {}),
        ):
            score, metadata, evidence = feature_diff.fused_evidence(
                self.reference,
                self.reference.copy(),
                require_cross_evidence=True,
            )
        self.assertEqual(metadata["absolute_signal_gate"], "suppressed_low_signal")
        self.assertEqual(metadata["fusion_policy"], "suppressed_low_absolute_signal")
        self.assertEqual(float(score.max()), 0.0)
        self.assertEqual(float(evidence["agreement"].max()), 0.0)

    def test_large_dino_signal_is_not_suppressed_by_the_gate(self) -> None:
        with patch.object(
            feature_diff,
            "dino_difference",
            return_value=(np.full((40, 60), 0.20, dtype=np.float32), {}),
        ):
            _score, metadata, _evidence = feature_diff.fused_evidence(
                self.reference,
                self.reference.copy(),
                require_cross_evidence=True,
            )
        self.assertEqual(metadata["absolute_signal_gate"], "passed")


if __name__ == "__main__":
    unittest.main()
