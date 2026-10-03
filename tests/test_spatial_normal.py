import unittest
import numpy as np
from tools.probe_spatial_normal import fit_spatial, spatial_score, normal_scale, fuse


class SpatialTests(unittest.TestCase):
    def test_constant_features_and_singular_covariance_stay_finite(self):
        bank = np.zeros((8,2,3,4)); bank[...,0] = 1
        model = fit_spatial(bank, dimensions=4)
        np.testing.assert_allclose(spatial_score(bank[0], model), 0, atol=1e-8)
        anomaly = bank[0].copy(); anomaly[0,0] = [0,1,0,0]
        score = spatial_score(anomaly, model)
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[0,0], 10)
        self.assertEqual(score[1,1], 0)

    def test_same_parts_in_wrong_positions_are_not_normal(self):
        bank = np.zeros((8,1,2,2)); bank[:,0,0,0] = 1; bank[:,0,1,1] = 1
        model = fit_spatial(bank, dimensions=2)
        self.assertTrue((spatial_score(bank[0,:,::-1], model) > 10).all())

    def test_regularization_seed_and_input_preservation(self):
        bank = np.random.default_rng(1).normal(size=(12,2,3,40))
        original = bank.copy()
        first, second = fit_spatial(bank), fit_spatial(bank)
        np.testing.assert_array_equal(bank, original)
        np.testing.assert_array_equal(first['channels'], second['channels'])
        np.testing.assert_allclose(first['precision'], second['precision'])
        self.assertTrue(np.isfinite(spatial_score(bank[0], first)).all())

    def test_rejects_invalid_features_and_geometry(self):
        with self.assertRaises(ValueError): fit_spatial(np.zeros((1,2,3,40)))
        with self.assertRaises(ValueError): fit_spatial(np.full((8,2,3,40), np.nan))
        with self.assertRaises(ValueError): fit_spatial(np.zeros((8,2,3,40)), shrinkage=0)
        model = fit_spatial(np.zeros((8,2,3,40)))
        with self.assertRaises(ValueError): spatial_score(np.zeros((1,3,40)), model)

    def test_normal_calibration_and_fusion_are_label_free(self):
        maps = [np.full((2,3), value) for value in (1,2,3,4)]
        scale, info = normal_scale(maps)
        self.assertAlmostEqual(scale, 3.85)
        self.assertFalse(info['floor_applied'])
        zero_scale, zero_info = normal_scale([np.zeros((2,3))]*3)
        self.assertEqual(zero_scale, 1e-8)
        self.assertTrue(zero_info['floor_applied'])
        np.testing.assert_allclose(fuse(maps[0], maps[1], 1, 1), maps[1])
        with self.assertRaises(ValueError): normal_scale([np.ones((2,3))])


if __name__ == '__main__':
    unittest.main()
