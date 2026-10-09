import copy,json,unittest
from pathlib import Path
import llm_review_priority as p

class PriorityTests(unittest.TestCase):
    def setUp(self):
        self.ids=['candidate_001']
        self.bound={'test':'fixture'}
        self.region={'candidate_id':self.ids[0],'observation_zh':'可见形状变化',
                     'hypotheses':['appearance_change'],'requested_checks':['inspect_raw_region'],
                     'question_zh':'请核对原图？','confidence':'medium','judgment':'visible_change','suggested_priority':'high'}
        self.result={'status':'ok','binding':self.bound,'plan':{'review_order':self.ids,'regions':[self.region],'summary_zh':'复核变化'}}
        self.evidence=[{'candidate_id':self.ids[0],'alignment_reliable':True,'mask_changed_pixels':10,
                       'strong_local_wire_evidence':True,'baseline_priority':'normal'}]
    def final(self):return p.adjust(self.result,self.evidence,self.bound)['rows'][0]['adjusted_priority']
    def test_supported_upgrade(self):self.assertEqual(self.final(),'high')
    def test_artifact_downgrade(self):
        self.region.update(judgment='likely_artifact',suggested_priority='low')
        self.evidence[0].update(strong_local_wire_evidence=False,mask_changed_pixels=0)
        self.assertEqual(self.final(),'low')
    def test_strong_evidence_blocks_downgrade(self):
        self.region.update(judgment='likely_artifact',suggested_priority='low')
        self.assertEqual(self.final(),'normal')
    def test_changed_mask_blocks_downgrade(self):
        self.region.update(judgment='likely_artifact',suggested_priority='low')
        self.evidence[0]['strong_local_wire_evidence']=False
        self.assertEqual(self.final(),'normal')
    def test_unreliable_alignment(self):
        self.evidence[0]['alignment_reliable']=False;self.assertEqual(self.final(),'normal')
    def test_low_confidence(self):
        self.region['confidence']='low';self.assertEqual(self.final(),'normal')
    def test_unknown_does_not_downgrade(self):
        self.region.update(judgment='insufficient_evidence',suggested_priority='normal')
        self.assertEqual(self.final(),'normal')
    def test_failure_keeps_priority(self):
        self.result['status']='fallback_local_review';self.assertEqual(self.final(),'normal')
    def test_stale_binding(self):
        self.result['binding']={};self.assertEqual(self.final(),'normal')
    def test_unexpected_electrical_verdict_rejected(self):
        self.region['electrical_verdict']='correct'
        self.assertEqual(self.final(),'normal')
    def test_no_mutation_or_removal(self):
        frozen=copy.deepcopy((self.result,self.evidence))
        result=p.adjust(self.result,self.evidence,self.bound)
        self.assertEqual((self.result,self.evidence),frozen)
        self.assertTrue(result['rows'][0]['candidate_retained'])
        self.assertEqual(result['rows'][0]['electrical_verdict'],'not_assessed')
    def test_missing_region(self):
        self.result['plan']['regions']=[];self.assertEqual(self.final(),'normal')
    def test_unknown_enum(self):
        for key,value in [('judgment','correct'),('suggested_priority','green')]:
            bad=copy.deepcopy(self.result['plan']);bad['regions'][0][key]=value
            with self.assertRaises(ValueError):p.validate(bad,self.ids)
    def test_actual_inputs(self):
        from probe_llm_recheck_planner_20261008 import CASES
        for path in CASES.values():
            report=json.loads(path.read_text(encoding='utf-8'));fusion=report['sam3_fusion']
            ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
            ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
            evidence=p.local_evidence(report,ref,ins)
            self.assertEqual(len(evidence),len(report['review_regions']))
            request,ids=p.payload(ref,ins,report['review_regions'],p.base.backend.load_settings())
            self.assertEqual(len(ids),len(evidence))
            self.assertNotIn(str(path),json.dumps(request))

if __name__=='__main__':unittest.main()
