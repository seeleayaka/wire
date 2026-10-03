import unittest
import numpy as np
from tools.prepare_fine_heat_evidence import mean_model
from tools.probe_spatial_normal import normalized_features


class FineMeanTests(unittest.TestCase):
    def test_streamed_mean_matches_normalized_reference_without_mutation(self):
        bank=np.random.default_rng(4).normal(size=(5,3,4,8)).astype(np.float32)
        before=bank.copy(); channels=np.array([0,3,6])
        model=mean_model(bank,channels)
        np.testing.assert_allclose(model['mean'],normalized_features(bank)[...,channels].mean(axis=0),atol=1e-12)
        np.testing.assert_array_equal(bank,before)
        np.testing.assert_array_equal(model['channels'],channels)

    def test_constant_bank_and_invalid_geometry(self):
        bank=np.zeros((4,2,3,8),dtype=np.float32)
        self.assertTrue(np.isfinite(mean_model(bank,np.array([1,4]))['mean']).all())
        with self.assertRaises(ValueError): mean_model(np.zeros((0,2,3,8)),np.array([1]))
        with self.assertRaises(ValueError): mean_model(bank,np.array([9]))


if __name__=='__main__':
    unittest.main()
