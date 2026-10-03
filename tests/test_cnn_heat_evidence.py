import unittest
import numpy as np
import torch
from tools.prepare_cnn_heat_evidence import combine_maps


class CnnEvidenceTests(unittest.TestCase):
    def test_pyramid_grid_equal_layer_norms_and_no_input_mutation(self):
        maps=[torch.ones(1,2,8,12),torch.ones(1,4,4,6)*10]
        before=[m.clone() for m in maps]
        result=combine_maps(maps,(3,5))
        self.assertEqual(result.shape,(3,5,6))
        np.testing.assert_allclose(np.linalg.norm(result[:,:,:2],axis=-1),1,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(result[:,:,2:],axis=-1),1,atol=1e-6)
        for value,old in zip(maps,before): self.assertTrue(torch.equal(value,old))

    def test_zero_features_finite_and_invalid_maps_rejected(self):
        maps=[torch.zeros(1,2,4,4),torch.zeros(1,4,2,2)]
        self.assertTrue(np.isfinite(combine_maps(maps,(3,3))).all())
        with self.assertRaises(ValueError): combine_maps(maps[:1],(3,3))
        with self.assertRaises(ValueError): combine_maps([torch.full((1,2,4,4),float('nan')),maps[1]],(3,3))


if __name__=='__main__':
    unittest.main()
