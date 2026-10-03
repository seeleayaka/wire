import copy,unittest
from unittest.mock import patch
from inspection_agent.consensus_port_rescue import expand_consensus_result,run_consensus_port_rescue

def packet():
    rows=[dict(left=30+i*40,top=30,right=50+i*40,bottom=50,class_id=0,confidence=.99-i*.01,valid_warp_fraction=1) for i in range(12)]
    native=[dict(box_xyxy=[p[k] for k in ('left','top','right','bottom')],class_id=0,confidence=p['confidence'],source_tile=0) for p in rows]
    raw=[{**p,'source_tile':tile} for p in native for tile in (0,1)]
    return dict(status='applied',parents=[dict(bbox_xyxy=[1,2,20,40])],existing_hints=[],
        rescue_hints=[dict(box=copy.deepcopy(p),role='original') for p in rows[:5]],
        analysis_rois=[dict(left=0,top=0,right=600,bottom=200)],
        source_evidence=dict(predictions=dict(source_shape=[1000,1000],merged_predictions=native,edge_kept_predictions=raw),aligned_predictions=rows),
        reference_evidence=dict(aligned_predictions=[]),automatic_fault_verdict=False)

class ConsensusTests(unittest.TestCase):
    def test_default_no_supplement(self):
        raw=packet();out=expand_consensus_result(raw)
        self.assertEqual(out['rescue_hints'],raw['rescue_hints']);self.assertEqual(out['supplementary_hints'],[])
    def test_five_plus_five_immutable(self):
        raw=packet();before=copy.deepcopy(raw);out=expand_consensus_result(raw,supplementary_enabled=True)
        self.assertEqual(raw,before);self.assertEqual(out['rescue_hints'],raw['rescue_hints'])
        self.assertEqual(len(out['supplementary_hints']),5);self.assertEqual(out['parents'],raw['parents'])
        self.assertFalse(out['automatic_fault_verdict'])
    def test_single_tile_rejected(self):
        raw=packet();raw['source_evidence']['predictions']['edge_kept_predictions']=[p for p in raw['source_evidence']['predictions']['edge_kept_predictions'] if p['source_tile']==0]
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    def test_weak_second_tile_rejected(self):
        raw=packet()
        for p in raw['source_evidence']['predictions']['edge_kept_predictions']:
            if p['source_tile']==1:p['confidence']=.5
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    def test_reference_still_suppresses(self):
        raw=packet();raw['reference_evidence']['aligned_predictions']=raw['source_evidence']['aligned_predictions']
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    def test_existing_hints_dedupe(self):
        raw=packet();raw['existing_hints']=[dict(box=p) for p in raw['source_evidence']['aligned_predictions']]
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    def test_roi_and_warp_unchanged(self):
        raw=packet();raw['analysis_rois']=[dict(left=0,top=0,right=20,bottom=20)]
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
        raw=packet()
        for p in raw['source_evidence']['aligned_predictions'][5:]:p['valid_warp_fraction']=.9
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    def test_native_frame_boundary(self):
        raw=packet()
        for p in raw['source_evidence']['predictions']['merged_predictions'][5:]:p['box_xyxy'][0]=0
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    def test_fallback_not_bypassed(self):
        raw=dict(status='fallback',rescue_hints=[],fallback_reason='unsupported_scene')
        self.assertEqual(expand_consensus_result(raw,supplementary_enabled=True)['supplementary_hints'],[])
    @patch('inspection_agent.consensus_port_rescue.run_precision_port_rescue')
    def test_default_off_api(self,primary):
        primary.return_value=dict(status='disabled',rescue_hints=[])
        self.assertEqual(run_consensus_port_rescue({},project='unused')['supplementary_hints'],[])
        primary.assert_called_once_with({},project='unused',enabled=False,scene='unknown')
    def test_count_mismatch_rejected(self):
        raw=packet();raw['source_evidence']['predictions']['merged_predictions']=[]
        with self.assertRaisesRegex(ValueError,'native_aligned_prediction_count_mismatch'):expand_consensus_result(raw,supplementary_enabled=True)
    @patch('inspection_agent.consensus_port_rescue.run_precision_port_rescue')
    def test_optional_failure_keeps_primary(self,primary):
        raw=packet();raw['source_evidence']['predictions']['merged_predictions']=[];primary.return_value=raw
        result=run_consensus_port_rescue({},project='unused',enabled=True,supplementary_enabled=True)
        self.assertEqual(result['rescue_hints'],raw['rescue_hints'])
        self.assertEqual(result['status'],'applied');self.assertEqual(result['supplementary_hints'],[])
        self.assertIn('native_aligned_prediction_count_mismatch',result['supplementary_policy']['fallback_reason'])

if __name__=='__main__':unittest.main()
