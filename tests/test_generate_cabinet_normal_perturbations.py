from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

from generate_cabinet_normal_perturbations import VARIANT_DESCRIPTIONS, generate_variants


class CabinetNormalPerturbationsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows, columns = np.indices((24, 32))
        self.reference = np.dstack(
            (
                (rows * 7 % 256).astype(np.uint8),
                (columns * 5 % 256).astype(np.uint8),
                ((rows + columns) * 3 % 256).astype(np.uint8),
            )
        )

    def test_variants_preserve_bgr_shape_and_are_repeatable(self) -> None:
        first = generate_variants(self.reference, seed=7)
        second = generate_variants(self.reference, seed=7)
        self.assertEqual(set(first), set(VARIANT_DESCRIPTIONS))
        for name, image in first.items():
            self.assertEqual(image.shape, self.reference.shape, name)
            self.assertEqual(image.dtype, np.uint8, name)
            self.assertTrue(np.array_equal(image, second[name]), name)

    def test_rejects_non_bgr_uint8_input(self) -> None:
        with self.assertRaises(ValueError):
            generate_variants(self.reference[:, :, 0])
        with self.assertRaises(ValueError):
            generate_variants(self.reference.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
