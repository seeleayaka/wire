import unittest
import numpy as np
from affine_anchor_holdout import fit_affine

class AffineHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([(x,y) for x in range(0,101,10) for y in range(0,101,10)],np.float32)
        self.anchor=dict(id='node',bbox_xyxy=[40,40,60,60])
    def fit(self,src,dst):return fit_affine(src,dst,self.anchor,[0,0,101,101],np.eye(3))
    def test_exact_affine_supported(self):
        r=self.fit(self.points+[3,7],self.points)
        self.assertTrue(r['localization_proposal_supported']);self.assertFalse(r['identity_verified'])
    def test_sparse_correspondences_abstain(self):self.assertFalse(self.fit(self.points[:8],self.points[:8])['localization_proposal_supported'])
    def test_nonfinite_rejected(self):
        p=self.points.copy();p[0,0]=np.nan
        with self.assertRaises(ValueError):self.fit(p,self.points)
    def test_missing_spatial_surrounding_abstains(self):
        p=self.points.copy();p[:,0]*=.2
        self.assertFalse(self.fit(p,p)['localization_proposal_supported'])
    def test_inputs_not_mutated(self):
        p=self.points.copy();self.fit(p,p);np.testing.assert_array_equal(p,self.points)
if __name__=='__main__':unittest.main()
