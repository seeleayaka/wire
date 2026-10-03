import copy,unittest
from unittest.mock import patch
from inspection_agent.context_port_recheck import (recheck_proposals,recheck_windows,confirm_rechecks,
    append_verified_rechecks,run_context_port_recheck)

def row(score=.9,box=None,cls=1):
    return dict(confidence=score,class_id=cls,box_xyxy=box or [1000,1000,1020,1040],source_tile=0,support_tiles=[0])
def raw(rows=None):return dict(source_shape=[2736,3648],merged_predictions=rows or [],edge_kept_predictions=rows or [])
def entry():
    seed=row(.7)
    return dict(proposal=seed,windows=recheck_windows(seed,[2736,3648]),views=[[row(.85)],[row(.8)]])
def result():return dict(status='applied',parents=[],existing_hints=[],rescue_hints=[],supplementary_hints=[],
    source_evidence={'predictions':raw()},reference_evidence={'predictions':raw(),'aligned_predictions':[]},
    analysis_rois=[dict(left=0,top=0,right=3648,bottom=2736)],recheck_policy={},supplementary_policy={})

class RecheckTests(unittest.TestCase):
    def test_strong_unsupported_only(self):
        rows=[row(.9,[200+i*220,1500,220+i*220,1540]) for i in range(7)]+[row(.2)]
        self.assertEqual(len(recheck_proposals(raw(rows))),2)
    def test_boundary_and_large_refused(self):
        self.assertEqual(recheck_proposals(raw([row(.5),row(.9,[1,100,21,140]),row(.9,[200,100,500,140])])),[])
    def test_two_strict_votes(self):self.assertEqual(len(confirm_rechecks([entry()],[2736,3648])),1)
    def test_threshold_not_inclusive(self):
        e=entry();e['views'][1][0]['confidence']=.75
        self.assertEqual(confirm_rechecks([e],[2736,3648]),[])
    def test_class_match(self):
        e=entry();e['views'][1][0]['class_id']=0
        self.assertEqual(confirm_rechecks([e],[2736,3648]),[])
    def test_same_crop_not_two_votes(self):
        e=entry();e['windows'][1]=e['windows'][0]
        self.assertEqual(confirm_rechecks([e],[2736,3648]),[])
    def test_append_without_input_mutation(self):
        r=result();before=copy.deepcopy(r);c=confirm_rechecks([entry()],[2736,3648])
        import numpy as np
        out=append_verified_rechecks(r,c,[],np.eye(3))
        self.assertEqual(r,before);self.assertEqual(len(out['supplementary_hints']),1)
        self.assertEqual(out['rescue_hints'],r['rescue_hints'])
    def test_static_reference_veto(self):
        r=result();c=confirm_rechecks([entry()],[2736,3648]);ref=dict(left=1000,top=1000,right=1020,bottom=1040,class_id=1,confidence=.3)
        import numpy as np
        self.assertEqual(append_verified_rechecks(r,c,[ref],np.eye(3))['supplementary_hints'],[])
    def test_outside_roi_refused(self):
        r=result();r['analysis_rois']=[dict(left=0,top=0,right=300,bottom=300)]
        import numpy as np
        self.assertEqual(append_verified_rechecks(r,confirm_rechecks([entry()],[2736,3648]),[],np.eye(3))['supplementary_hints'],[])
    def test_extra_budget_full(self):
        r=result();r['supplementary_hints']=[{'box':dict(left=200+i*50,top=1500,right=220+i*50,bottom=1540)} for i in range(5)]
        import numpy as np
        out=append_verified_rechecks(r,confirm_rechecks([entry()],[2736,3648]),[],np.eye(3))
        self.assertEqual(out['supplementary_hints'],r['supplementary_hints'])
    def test_disabled_never_loads_model(self):
        r=dict(status='disabled',rescue_hints=[],supplementary_hints=[])
        with patch('inspection_agent.context_port_recheck.run_consensus_port_rescue',return_value=r):
            out=run_context_port_recheck({},project='missing')
        self.assertEqual(out['status'],'disabled');self.assertFalse(out['recheck_policy']['enabled'])
    def test_optional_failure_preserves_original_cues(self):
        r=result();r['rescue_hints']=[{'box':{'left':30,'top':30,'right':50,'bottom':50}}];before=copy.deepcopy(r)
        with patch('inspection_agent.context_port_recheck.run_consensus_port_rescue',return_value=r),patch('inspection_agent.context_port_recheck.recheck_proposals',side_effect=ValueError('test')):
            out=run_context_port_recheck({},project='missing',enabled=True,supplementary_enabled=True)
        self.assertEqual(out['rescue_hints'],before['rescue_hints']);self.assertIn('test',out['recheck_policy']['fallback_reason'])

if __name__=='__main__':unittest.main()
