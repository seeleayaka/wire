import unittest
import torch
from reverse_pair_semantics import reverse_features, augment


class ReversePairTests(unittest.TestCase):
    def features(self):
        a = torch.arange(4*1536).float().reshape(4, 1536) / 10000
        b = a.flip(1)
        return torch.cat((a, b, (a-b).abs(), a*b), dim=1)

    def test_involution_and_symmetric_blocks_unchanged(self):
        x = self.features()
        r = reverse_features(x)
        self.assertTrue(torch.equal(reverse_features(r), x))
        self.assertTrue(torch.equal(r[:, 3072:], x[:, 3072:]))
        self.assertTrue(torch.equal(r[:, :1536], x[:, 1536:3072]))

    def test_original_rows_labels_and_folds_preserved(self):
        x = self.features()
        y, f = torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, 2, 1])
        before = x.clone()
        xx, yy, ff, ii = augment(x, y, f)
        self.assertEqual(ii.tolist(), [1, 2])
        self.assertTrue(torch.equal(xx[:4], x))
        self.assertTrue(torch.equal(yy, torch.tensor([0, 1, 2, 0, 0, 0])))
        self.assertTrue(torch.equal(ff, torch.tensor([0, 1, 2, 1, 1, 2])))
        self.assertTrue(torch.equal(x, before))

    def test_reverse_twins_cannot_cross_source_fold(self):
        xx, yy, ff, ii = augment(self.features(), torch.tensor([0, 1, 2, 0]), torch.tensor([2, 0, 1, 2]))
        for fold in range(3):
            retained = ff != fold
            for j, original in enumerate(ii.tolist()):
                self.assertEqual(bool(retained[original]), bool(retained[4+j]))

    def test_reference_self_is_fixed_point(self):
        a = torch.ones(2, 1536)
        x = torch.cat((a, a, torch.zeros_like(a), a*a), dim=1)
        self.assertTrue(torch.equal(reverse_features(x), x))

    def test_invalid_dimension_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            reverse_features(torch.ones(4, 12))
        x = self.features()
        x[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            reverse_features(x)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            augment(self.features(), torch.tensor([0, 3, 2, 0]), torch.tensor([0, 0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
