import copy
import unittest
from unittest.mock import patch
from inspection_agent.precision_port_rescue import precision_verified_result, run_precision_port_rescue, POLICY_ID

def p(left, score=.9):
    return dict(left=left+20,top=20,right=left+30,bottom=30,class_id=0,confidence=score,valid_warp_fraction=1)

def verify(raw):
    original=copy.deepcopy(raw)
    if original['status']=='applied':
        rows=original['source_evidence']['aligned_predictions']
        original['source_evidence']['predictions']=dict(source_shape=[200,400],
            merged_predictions=[dict(box_xyxy=[p[k] for k in ('left','top','right','bottom')],
                class_id=p['class_id'],confidence=p['confidence']) for p in rows])
    return precision_verified_result(original)

def packet():
    return dict(status='applied',parents=[{'bbox_xyxy':[1,2,3,4]}],existing_hints=[],rescue_hints=[],
        analysis_rois=[dict(left=0,top=0,right=200,bottom=100)],
        source_evidence={'aligned_predictions':[p(i*20) for i in range(8)]},
        reference_evidence={'aligned_predictions':[]},automatic_fault_verdict=False)

class PrecisionTests(unittest.TestCase):
    def test_maximum_five_preserve_inputs(self):
        raw=packet();before=copy.deepcopy(raw);output=verify(raw)
        self.assertEqual(len(output['rescue_hints']),5)
        self.assertEqual(raw,before);self.assertEqual(output['parents'],raw['parents'])
        self.assertFalse(output['automatic_fault_verdict'])
    def test_weak_first_removed(self):
        raw=packet();raw['source_evidence']['aligned_predictions']=[p(0,.3)]
        raw['rescue_hints']=[dict(box=p(0,.3),parent_index=None)]
        output=verify(raw)
        self.assertEqual(output['rescue_hints'],[])
        self.assertEqual(output['budget_policy']['suppressed_weak_hints'],1)
    def test_strict_threshold(self):
        raw=packet();raw['source_evidence']['aligned_predictions']=[p(0,.5)]
        raw['rescue_hints']=[dict(box=p(0,.5),parent_index=None)]
        self.assertEqual(verify(raw)['rescue_hints'],[])
    def test_low_first_cannot_block_high(self):
        raw=packet();raw['rescue_hints']=[dict(box=p(160,.3),parent_index=None)]
        output=verify(raw)
        self.assertEqual(len(output['rescue_hints']),5)
        self.assertTrue(all(h['box']['confidence']>.5 for h in output['rescue_hints']))
    def test_reference_still_suppresses(self):
        raw=packet();raw['reference_evidence']['aligned_predictions']=copy.deepcopy(raw['source_evidence']['aligned_predictions'])
        self.assertEqual(verify(raw)['rescue_hints'],[])
    def test_existing_hints_unchanged(self):
        raw=packet();raw['existing_hints']=[dict(box=p(i*20,.3)) for i in range(8)]
        output=verify(raw)
        self.assertEqual(output['existing_hints'],raw['existing_hints'])
        self.assertEqual(output['rescue_hints'],[])
    def test_geometry_still_required(self):
        raw=packet();weak=p(0);weak['valid_warp_fraction']=.9
        raw['source_evidence']['aligned_predictions']=[p(220),weak]
        self.assertEqual(verify(raw)['rescue_hints'],[])
    def test_fallback_preserved(self):
        raw=dict(status='fallback',rescue_hints=[],fallback_reason='local_alignment_not_supported')
        output=verify(raw)
        self.assertEqual(output['rescue_hints'],[])
        self.assertEqual(output['fallback_reason'],raw['fallback_reason'])
    @patch('inspection_agent.precision_port_rescue.run_independent_port_rescue')
    def test_default_off_forwarded(self, backend):
        backend.return_value=dict(status='disabled',rescue_hints=[])
        output=run_precision_port_rescue({'inspection':'unused'},project='unused')
        backend.assert_called_once_with({'inspection':'unused'},project='unused',enabled=False,scene='unknown')
        self.assertEqual(output['status'],'disabled')
        self.assertEqual(output['budget_policy']['policy_id'],POLICY_ID)
    def test_all_four_native_edges(self):
        for box in ([0,20,10,30],[20,0,30,10],[390,20,400,30],[20,190,30,200]):
            raw=packet();row=p(0)
            row.update(zip(('left','top','right','bottom'),box))
            raw['source_evidence']['aligned_predictions']=[row]
            self.assertEqual(verify(raw)['rescue_hints'],[])
    def test_invalid_native_pairing_rejected(self):
        raw=packet();raw['source_evidence']['predictions']=dict(source_shape=[200,400],merged_predictions=[])
        with self.assertRaisesRegex(ValueError,'native_aligned_prediction_count_mismatch'):
            precision_verified_result(raw)
    def test_evidence_stays_unfiltered(self):
        raw=packet();raw['source_evidence']['aligned_predictions'][0]['left']=0
        output=verify(raw)
        self.assertEqual(output['source_evidence']['aligned_predictions'],raw['source_evidence']['aligned_predictions'])
        self.assertGreater(output['budget_policy']['suppressed_native_edge_candidates'],0)
    def test_native_edge_cannot_hide_after_alignment(self):
        raw=packet();raw['source_evidence']['aligned_predictions']=[p(80)]
        raw['source_evidence']['predictions']=dict(source_shape=[200,400],
            merged_predictions=[dict(box_xyxy=[0,20,10,30],class_id=0,confidence=.9)])
        self.assertEqual(precision_verified_result(raw)['rescue_hints'],[])
    def test_no_annotation_or_image_name_dependency(self):
        raw=packet();before=verify(raw)
        raw['label']='different';raw['source_image_name']='not_a_selected_sample.JPG'
        self.assertEqual(verify(raw)['rescue_hints'],before['rescue_hints'])

if __name__=='__main__':unittest.main()
