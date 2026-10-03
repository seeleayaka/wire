import unittest
import numpy as np
from tools.probe_spatial_metric import unweighted_score
from tools.probe_spatial_normal import spatial_score


class MetricTests(unittest.TestCase):
    def test_identity_precision_matches_unweighted_metric(self):
        query=np.random.default_rng(2).normal(size=(2,3,8))
        model={'channels':np.array([1,3,5]),'mean':np.zeros((2,3,3)),
               'precision':np.broadcast_to(np.eye(3),(2,3,3,3))}
        np.testing.assert_allclose(unweighted_score(query,model),spatial_score(query,model))

    def test_preserves_input_and_rejects_geometry_errors(self):
        query=np.ones((2,3,8)); before=query.copy()
        model={'channels':np.array([0,2]),'mean':np.zeros((2,3,2))}
        self.assertTrue(np.isfinite(unweighted_score(query,model)).all())
        np.testing.assert_array_equal(query,before)
        with self.assertRaises(ValueError): unweighted_score(np.ones((1,3,8)),model)


if __name__=='__main__':
    unittest.main()
