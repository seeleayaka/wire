import copy,json,unittest
from pathlib import Path
import llm_review_priority as risk
from probe_llm_recheck_planner_20261008 import ROOT,CASES

class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        report=json.loads(CASES['cabinet2'].read_text(encoding='utf-8'));fusion=report['sam3_fusion']
        self.candidates=report['review_regions']
        self.ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
        self.ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
        self.fixture=json.loads((ROOT/'artifacts/llm_priority_cabinet2_diagnostic_20261008/response_fixture.json').read_text(encoding='utf-8'))
    def invoke(self,plan,finish='stop'):
        return risk.run(self.ref,self.ins,self.candidates,api_key='fake-test-key',
                        sender=lambda *_:{'model':'mock','choices':[{'finish_reason':finish,'message':{'content':json.dumps(plan)}}]})
    def test_actual_fixture_passes_without_request(self):
        self.assertEqual(self.invoke(self.fixture)['status'],'ok')
    def test_rejection_reason_preserved(self):
        bad=copy.deepcopy(self.fixture);bad['regions'][0]['confidence']='high'
        result=self.invoke(bad)
        self.assertEqual(result['status'],'fallback_local_review')
        self.assertEqual(result['diagnostic']['failure_stage'],'validation')
        self.assertEqual(result['diagnostic']['reason'],'Unsupported confidence')
    def test_unknown_judgment(self):
        bad=copy.deepcopy(self.fixture);bad['regions'][0]['judgment']='PRIVATE_EXTERNAL_VALUE'
        result=self.invoke(bad)
        self.assertEqual(result['diagnostic']['reason'],'Invalid judgment or priority')
        self.assertNotIn('PRIVATE_EXTERNAL_VALUE',json.dumps(result))
    def test_sensitive_exception_not_saved(self):
        def fail(*_):raise OSError('fake-test-key PRIVATE_EXTERNAL_BODY')
        result=risk.run(self.ref,self.ins,self.candidates,api_key='fake-test-key',sender=fail)
        self.assertEqual(result['diagnostic']['failure_stage'],'transport')
        self.assertNotIn('fake-test-key',json.dumps(result))
        self.assertNotIn('PRIVATE_EXTERNAL_BODY',json.dumps(result))

if __name__=='__main__':unittest.main()
