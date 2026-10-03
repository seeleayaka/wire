import unittest
from itertools import product
from pathlib import Path
from tools.audit_cnn_eligibility import support_facts


class EligibilityTests(unittest.TestCase):
    def box(self,whole=False,primary=1,refine=0,cross=.9,thin=False):
        return {'evidence_summary':{'whole_roi_overlap':whole,'primary_tile_count':primary,
                'refinement_tile_count':refine,'thin_core_observation_count':int(thin),
                'merged_observation_count':1,'tile_edge_only':False},
                'evidence_scores':{'cross_evidence_pixel_ratio_max':cross}}
    def test_original_support_branches(self):
        self.assertTrue(support_facts(self.box(whole=True,cross=.1))['allowed'])
        self.assertTrue(support_facts(self.box(primary=4,cross=.1))['allowed'])
        self.assertTrue(support_facts(self.box(refine=1,cross=.5))['allowed'])
        self.assertTrue(support_facts(self.box(primary=0,refine=2))['allowed'])
    def test_rejection_reasons_and_thin_stricter_gate(self):
        self.assertEqual(support_facts(self.box())['reason'],'insufficient_spatial_support')
        self.assertEqual(support_facts(self.box(refine=1,cross=.499))['reason'],'weak_cross_evidence')
        self.assertEqual(support_facts(self.box(whole=True,thin=True,cross=.1))['reason'],'thin_missing_cross_evidence')
        self.assertEqual(support_facts(self.box(whole=True,thin=True))['reason'],'thin_missing_cross_scale')
        self.assertTrue(support_facts(self.box(whole=True,refine=1,thin=True))['allowed'])
    def test_matches_real_production_gate_over_support_combinations(self):
        from tools.merge_audit import setup
        _,tiled=setup(Path(__file__).resolve().parents[1])
        for whole,primary,refine,cross,thin in product((False,True),(0,1,3,4),(0,1,2),(.49,.50,.9),(False,True)):
            box=self.box(whole,primary,refine,cross,thin)
            displayed,_=tiled._display_candidates_for_large_roi([box])
            self.assertEqual(support_facts(box)['allowed'],bool(displayed))


if __name__=='__main__': unittest.main()
