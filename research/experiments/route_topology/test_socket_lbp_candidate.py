import unittest
import numpy as np
from socket_lbp_candidate import descriptor, resolve

class TextureTests(unittest.TestCase):
    def test_descriptor_finite_normalized_and_input_unchanged(self):
        rgb = np.random.default_rng(0).integers(0, 256, (50, 100, 3), dtype=np.uint8)
        before = rgb.copy()
        f = descriptor(rgb)
        self.assertEqual(f.shape, (160,))
        np.testing.assert_allclose(f.reshape(16,10).sum(1), 1)
        np.testing.assert_array_equal(rgb, before)
        np.testing.assert_array_equal(f, descriptor(rgb))

    def test_unstable_or_unknown_probes_cannot_resolve(self):
        for labels in [[1]*8+[None], [1]*8+[0], [None]*9]:
            self.assertIsNone(resolve(None, labels))
        self.assertEqual(resolve(None, [1]*9), 1)
        self.assertEqual(resolve(None, [0]*9), 0)

    def test_prior_supported_results_never_replaced(self):
        self.assertEqual(resolve(0, [1]*9), 0)
        self.assertEqual(resolve(1, [0]*9), 1)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError): descriptor(np.zeros((50,100,3)))
        with self.assertRaises(ValueError): descriptor(np.zeros((51,100,3), dtype=np.uint8))
        with self.assertRaises(ValueError): resolve(None, [1]*8)

if __name__ == '__main__': unittest.main()
