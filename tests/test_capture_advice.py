import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'prototype'))
from capture_advice import build_capture_advice,render_capture_advice

class CaptureAdviceTests(unittest.TestCase):
    def test_bad_alignment_prioritizes_same_view(self):
        value=build_capture_advice({'alignment':{'alignment_quality':{'reliable':False}}})
        self.assertEqual(value['comparison_view'],'same_as_reference')
        self.assertEqual(value['items'][0]['code'],'retake_same_view')
        self.assertEqual(value['alignment_state'],'unreliable')
    def test_missing_alignment_is_unknown_not_reliable(self):
        self.assertEqual(build_capture_advice({})['alignment_state'],'unknown')
    def test_top_level_bad_quality_overrides_nested_good(self):
        value=build_capture_advice({'alignment_quality':{'reliable':False},
                                   'alignment':{'alignment_quality':{'reliable':True}}})
        self.assertEqual(value['alignment_state'],'unreliable')
    def test_candidates_offer_local_and_supplemental_angle(self):
        value=build_capture_advice({'alignment':{'alignment_quality':{'reliable':True}},
                                   'review_regions':[{'id':999},{'tier':'dino_only'}]})
        by={r['code']:r for r in value['items']}
        self.assertIn('closeup_candidates',by);self.assertIn('supplemental_angle',by)
        self.assertIn('不能',by['supplemental_angle']['text'])
        self.assertEqual(by['closeup_candidates']['candidate_ids'],['candidate_001','candidate_002'])
        self.assertIn('如',by['supplemental_angle']['text'])
    def test_no_candidates_does_not_claim_electrical_pass(self):
        value=build_capture_advice({'decision':'no_significant_wire_related_difference','review_regions':[]})
        self.assertFalse(value['electrical_verdict_assessed'])
        self.assertIn('不等于',render_capture_advice(value))
    def test_sam_error_does_not_infer_disconnection(self):
        value=build_capture_advice({'sam3_fusion':{'status':'error'}})
        self.assertIn('sam_unavailable',[r['code'] for r in value['items']])
        self.assertNotIn('已断线',render_capture_advice(value))
    def test_does_not_mutate_report_or_depend_on_llm(self):
        report={'review_regions':[{}],'deepseek_mask_review':{'status':'error'},'decision':'manual_review'}
        original=copy.deepcopy(report);a=build_capture_advice(report)
        report['deepseek_mask_review']={'status':'ok','plan':{'summary':'model guesses'}}
        self.assertEqual(a,build_capture_advice(report))
        report['deepseek_mask_review']=original['deepseek_mask_review'];self.assertEqual(report,original)
        self.assertFalse(a['requires_llm']);self.assertEqual(a['network_requests'],0)
    def test_text_is_plain_and_safety_present(self):
        text=render_capture_advice(build_capture_advice({'review_regions':[{}]}))
        self.assertIn('不移动',text);self.assertIn('拍摄建议',text)

if __name__=='__main__':unittest.main()
