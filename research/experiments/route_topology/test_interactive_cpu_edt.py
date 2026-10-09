import unittest
import numpy as np
import torch
from interactive_cpu_edt import edt_cpu


class TestCPUEDT(unittest.TestCase):
    def test_brute_force_distances(self):
        rng = np.random.default_rng(17)
        for _ in range(20):
            active = rng.integers(0, 2, (2, 7, 9), dtype=np.uint8)
            result = edt_cpu(torch.from_numpy(active)).numpy()
            for i in range(2):
                zeros = np.argwhere(active[i] == 0)
                expected = np.array([[np.sqrt(((zeros - [y, x]) ** 2).sum(1).min())
                                      for x in range(9)] for y in range(7)])
                np.testing.assert_allclose(result[i], expected, atol=1e-6)

    def test_degenerate_and_unchanged_input(self):
        data = torch.stack([torch.zeros(3, 5), torch.ones(3, 5)])
        original = data.clone()
        result = edt_cpu(data)
        self.assertTrue(torch.equal(data, original))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue((result[0] == 0).all())
        self.assertTrue((result[1] == 1e9).all())

    def test_bad_rank_rejected(self):
        with self.assertRaises(ValueError):
            edt_cpu(torch.zeros(2, 3))


if __name__ == '__main__':
    unittest.main()
