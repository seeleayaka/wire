import unittest
from tools.probe_cnn_qualified_support import qualifies


class QualifiedSupportTests(unittest.TestCase):
    def box(self,primary=1,refine=1,cross=.1,thin=False,whole=False):
        return {'evidence_summary':{'whole_roi_overlap':whole,'primary_tile_count':primary,
                'refinement_tile_count':refine,'thin_core_observation_count':int(thin),
                'merged_observation_count':1,'tile_edge_only':False},
                'evidence_scores':{'cross_evidence_pixel_ratio_max':cross}}
    def test_old_eligibility_retained_even_below_cnn_threshold(self):
        self.assertTrue(qualifies(self.box(whole=True),0,3))
        self.assertTrue(qualifies(self.box(primary=4,refine=0),0,3))
    def test_cnn_exception_strict_threshold_and_spatial_support(self):
        self.assertTrue(qualifies(self.box(),3.01,3))
        self.assertFalse(qualifies(self.box(),3,3))
        self.assertFalse(qualifies(self.box(refine=0),100,3))
        self.assertTrue(qualifies(self.box(primary=0,refine=2),4,3))
    def test_thin_preserves_cross_scale_and_invalid_score_fails(self):
        self.assertFalse(qualifies(self.box(primary=0,refine=2,thin=True),4,3))
        self.assertTrue(qualifies(self.box(thin=True),4,3))
        with self.assertRaises(ValueError): qualifies(self.box(),float('nan'),3)
        with self.assertRaises(ValueError): qualifies(self.box(),4,-1)


if __name__=='__main__': unittest.main()
