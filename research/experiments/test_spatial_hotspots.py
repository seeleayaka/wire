import unittest
import numpy as np
from tools.diagnose_spatial_hotspots import top_mask,overlap_fraction,rank_correlation


class HotspotTests(unittest.TestCase):
    def test_exact_budget_and_deterministic_ties(self):
        score=np.ones((3,4))
        mask=top_mask(score,0.2)
        self.assertEqual(mask.sum(),3)
        np.testing.assert_array_equal(mask.ravel()[:3],True)
        self.assertFalse(mask.ravel()[3:].any())

    def test_overlap_and_border_do_not_wrap(self):
        score=np.arange(12).reshape(3,4)
        hot=top_mask(score,0.1)
        region=np.zeros((3,4),bool); region[2,3]=True
        self.assertEqual(overlap_fraction(hot,region),0.5)
        with self.assertRaises(ValueError): overlap_fraction(hot,np.zeros((2,4),bool))
        with self.assertRaises(ValueError): top_mask(np.full((3,4),np.nan))

    def test_tie_aware_rank_correlation_and_constant_maps(self):
        score=np.array([[1,1,3,5]])
        self.assertAlmostEqual(rank_correlation(score,score),1)
        self.assertAlmostEqual(rank_correlation(score,-score),-1)
        self.assertIsNone(rank_correlation(score,np.ones_like(score)))


if __name__=='__main__':
    unittest.main()
