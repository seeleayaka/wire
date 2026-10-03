import copy
import unittest
import numpy as np
from inspection_agent.experimental_cnn import select_with_optional_cnn


class ExperimentalCnnTests(unittest.TestCase):
    def box(self,x,whole=True):
        return dict(left=x,top=0,right=x+10,bottom=10,
                    evidence_summary={'whole_roi_overlap':whole,'primary_tile_count':1,'refinement_tile_count':1,
                                      'thin_core_observation_count':0,'merged_observation_count':1,'tile_edge_only':False},
                    evidence_scores={'cross_evidence_pixel_ratio_max':.1})
    def test_disabled_does_not_call_provider_and_preserves_data(self):
        base=[self.box(0)]; before=copy.deepcopy(base)
        def forbidden(): raise AssertionError('provider invoked while disabled')
        result,meta=select_with_optional_cnn(base,[],100,100,forbidden)
        self.assertEqual(result,before); self.assertEqual(meta['status'],'disabled')
        result[0]['left']=99; self.assertEqual(base,before)
    def test_exception_nan_shape_and_missing_provider_fall_back(self):
        base=[self.box(0)]
        def broken(): raise RuntimeError('missing cache')
        for provider in (broken,None,lambda:(np.full((21,28),np.nan),3,{}),lambda:(np.ones((2,2)),3,{})):
            result,meta=select_with_optional_cnn(base,base,100,100,provider,enabled=True)
            self.assertEqual(result,base); self.assertEqual(meta['status'],'fallback')
    def test_applied_budget_anchor_and_no_input_mutation(self):
        base=[self.box(0),self.box(20)]; novel=self.box(60,whole=False); pool=base+[novel]
        before=copy.deepcopy(pool); score=np.ones((21,28)); score[:3,16:20]=10
        result,meta=select_with_optional_cnn(base,pool,100,100,lambda:(score,3,{'frozen':True}),enabled=True)
        self.assertEqual(len(result),2); self.assertEqual(result[0],base[0]); self.assertEqual(result[1],novel)
        self.assertEqual(meta['status'],'applied'); self.assertEqual(pool,before)
    def test_invalid_geometry_or_missing_baseline_never_drops_old_candidates(self):
        base=[self.box(0)]
        for pool in ([],[dict(base[0],right=101)]):
            result,meta=select_with_optional_cnn(base,pool,100,100,lambda:(np.ones((21,28)),3,{}),enabled=True)
            self.assertEqual(result,base); self.assertEqual(meta['status'],'fallback')
    def test_empty_baseline_does_not_load_model(self):
        result,meta=select_with_optional_cnn([],[],100,100,None,enabled=True)
        self.assertEqual(result,[]); self.assertEqual(meta['status'],'skipped_empty_baseline')


if __name__=='__main__': unittest.main()
