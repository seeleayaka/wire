import copy, json, unittest
from dataclasses import replace
from pathlib import Path
import llm_recheck_planner as p

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/local_performance_fresh_20261007/cabinet_2/desktop_output/20261007_200359/report.json'

class PlannerTests(unittest.TestCase):
    def setUp(self):
        report=json.loads(REPORT.read_text(encoding='utf-8'))
        fusion=report['sam3_fusion']
        self.ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
        self.ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
        self.candidates=report['review_regions']
        self.settings=p.backend.load_settings()
        self.ids=['candidate_001','candidate_002']
        self.plan={'review_order':self.ids,'regions':[{'candidate_id':i,'observation_zh':'局部形状需核查',
                     'hypotheses':['insufficient_evidence'],'requested_checks':['inspect_raw_region'],
                     'question_zh':'请核对本地原图？','confidence':'low'} for i in self.ids],'summary_zh':'逐框核查并补证'}
    def sender(self,request,timeout):
        return {'model':'mock-only','choices':[{'message':{'content':json.dumps(self.plan,ensure_ascii=False)}}]}
    def test_valid(self): self.assertEqual(p.validate(self.plan,self.ids),self.plan)
    def test_closed_contract(self):
        for key in ('electrical_verdict','new_boxes','tool_calls'):
            bad=copy.deepcopy(self.plan);bad[key]=True
            with self.assertRaises(ValueError): p.validate(bad,self.ids)
    def test_ids(self):
        for ids in (['candidate_001'] ,['candidate_001']*2,['candidate_001','unknown']):
            bad=copy.deepcopy(self.plan);bad['review_order']=ids
            with self.assertRaises(ValueError): p.validate(bad,self.ids)
    def test_actions(self):
        for action in ('delete_box','mark_correct','execute_command','modify_threshold'):
            bad=copy.deepcopy(self.plan);bad['regions'][0]['requested_checks']=[action]
            with self.assertRaises(ValueError): p.validate(bad,self.ids)
    def test_region_ids(self):
        bad=copy.deepcopy(self.plan);bad['regions'][1]['candidate_id']=self.ids[0]
        with self.assertRaises(ValueError): p.validate(bad,self.ids)
    def test_high_confidence(self):
        bad=copy.deepcopy(self.plan);bad['regions'][0]['confidence']='high'
        with self.assertRaises(ValueError): p.validate(bad,self.ids)
    def test_incomplete(self):
        bad=copy.deepcopy(self.plan);bad['regions'].pop()
        with self.assertRaises(ValueError): p.validate(bad,self.ids)
    def test_isolation(self):
        before=copy.deepcopy(self.candidates)
        result=p.run(self.ref,self.ins,self.candidates,self.settings,'mock-secret',self.sender)
        self.assertEqual(result['status'],'ok');self.assertEqual(before,self.candidates)
        self.assertNotIn('mock-secret',json.dumps(result));self.assertEqual(result['automatic_actions_executed'],0)
    def test_payload_minimization(self):
        candidates=copy.deepcopy(self.candidates);candidates[0].update(raw_path='PRIVATE_PATH',ocr='PRIVATE_OCR',device='PRIVATE_DEVICE')
        request,_=p.payload(self.ref,self.ins,candidates,self.settings)
        text=json.dumps(request)
        for private in ('PRIVATE_PATH','PRIVATE_OCR','PRIVATE_DEVICE'): self.assertNotIn(private,text)
    def test_empty_no_network(self):
        def forbidden(*args): raise AssertionError('Unexpected network')
        result=p.run(self.ref,self.ins,[],self.settings,None,forbidden)
        self.assertEqual(result['status'],'no_candidates');self.assertEqual(result['network_requests'],0)
    def test_bad_response_fallback(self):
        result=p.run(self.ref,self.ins,self.candidates,self.settings,'mock',lambda *_:{'choices':[]})
        self.assertEqual(result['status'],'fallback_local_review');self.assertIsNone(result['plan'])
    def test_disabled_no_network(self):
        result=p.run(self.ref,self.ins,self.candidates,replace(self.settings,enabled=False),'mock',lambda *_:self.fail('network'))
        self.assertEqual(result['network_requests'],0)

if __name__=='__main__': unittest.main()
