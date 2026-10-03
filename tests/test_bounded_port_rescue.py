import unittest,copy
from inspection_agent.bounded_port_rescue import expand_verified_result
def p(l,score=.9):return dict(left=l,top=20,right=l+10,bottom=30,class_id=0,confidence=score,valid_warp_fraction=1)
def result():return dict(status='applied',parents=[{'bbox_xyxy':[1,2,3,4]}],existing_hints=[],rescue_hints=[],
    analysis_rois=[dict(left=0,top=0,right=200,bottom=100)],source_evidence={'aligned_predictions':[p(i*20) for i in range(8)]},reference_evidence={'aligned_predictions':[]})
class BoundedTests(unittest.TestCase):
    def test_max_five_and_preserve_inputs(self):
        raw=result();before=copy.deepcopy(raw);out=expand_verified_result(raw)
        self.assertEqual(len(out['rescue_hints']),5);self.assertEqual(raw,before)
        self.assertEqual(out['parents'],raw['parents'])
    def test_no_safety_gate_bypass(self):
        raw={'status':'fallback','rescue_hints':[],'fallback_reason':'local_alignment_not_supported'}
        self.assertEqual(expand_verified_result(raw)['rescue_hints'],[])
    def test_keep_lower_confidence_baseline(self):
        raw=result();hint=dict(box=p(0,.3),parent_index=None,automatic_fault_verdict=False)
        raw['rescue_hints']=[hint];out=expand_verified_result(raw)
        self.assertEqual(out['rescue_hints'][0],hint);self.assertEqual(len(out['rescue_hints']),5)
    def test_reject_reference_matches(self):
        raw=result();raw['reference_evidence']['aligned_predictions']=copy.deepcopy(raw['source_evidence']['aligned_predictions'])
        self.assertEqual(expand_verified_result(raw)['rescue_hints'],[])
    def test_roi_and_warp_still_required(self):
        raw=result();bad=p(250);weak=p(0);weak['valid_warp_fraction']=.9
        raw['source_evidence']['aligned_predictions']=[bad,weak,p(20,.5)]
        self.assertEqual(expand_verified_result(raw)['rescue_hints'],[])
    def test_existing_hints_dedupe(self):
        raw=result();raw['existing_hints']=[dict(box=p(i*20)) for i in range(8)]
        self.assertEqual(expand_verified_result(raw)['rescue_hints'],[])
if __name__=='__main__':unittest.main()
