import copy,unittest
from unittest.mock import patch
import numpy as np
from test_port_residual_feature_backend import base,cue
from port_resolution_backend import append_resolution_hints,run_resolution_review,POLICY_ID,WEIGHT_SHA

class ResolutionBackendTests(unittest.TestCase):
    def test_provenance_input_and_old_hints_preserved(self):
        old=base();before=copy.deepcopy(old);output,added=append_resolution_hints(old,[cue()],[],np.eye(3))
        self.assertEqual(old,before);self.assertEqual(len(added),1)
        self.assertEqual(output['rescue_hints'],old['rescue_hints'])
        self.assertEqual(added[0]['residual_policy_id'],POLICY_ID)
        self.assertEqual(added[0]['inference_imgsz'],1280)
        self.assertEqual(added[0]['student_weight_sha256'],WEIGHT_SHA)
        self.assertEqual(added[0]['evidence_tier'],'resolution_residual_manual_review')
    def test_reference_veto(self):
        ref=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_resolution_hints(base(),[cue()],[ref],np.eye(3))[1],[])
    def test_roi_and_warp(self):
        old=base();old['analysis_rois']=[dict(left=500,top=500,right=900,bottom=900)]
        self.assertEqual(append_resolution_hints(old,[cue()],[],np.eye(3))[1],[])
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_resolution_hints(base(),[cue()],[],matrix)[1],[])
    def test_existing_duplicate_cue_is_not_added(self):
        old=base();old['supplementary_hints']=[dict(box=dict(left=100,top=100,right=140,bottom=140))]
        self.assertEqual(append_resolution_hints(old,[cue()],[],np.eye(3))[1],[])
    def test_shared_budget_preserved(self):
        old=base();old['supplementary_hints']=[dict(box=dict(left=700+i*80,top=900,right=750+i*80,bottom=950)) for i in range(5)]
        out,added=append_resolution_hints(old,[cue()],[],np.eye(3))
        self.assertEqual(added,[]);self.assertEqual(out['supplementary_hints'],old['supplementary_hints'])
    def test_disabled_no_new_reads(self):
        with patch('port_resolution_backend.run_feature_residual_review',return_value=base()),patch('port_resolution_backend.sha',side_effect=RuntimeError('unexpected read')):
            self.assertFalse(run_resolution_review({})['resolution_policy']['enabled'])
    def test_fallback_preserves_original(self):
        old=base();old['status']='fallback'
        with patch('port_resolution_backend.run_feature_residual_review',return_value=old),patch('port_resolution_backend.sha',side_effect=RuntimeError('unexpected read')):
            self.assertEqual(run_resolution_review({},resolution_enabled=True)['status'],'fallback')
    def test_missing_branch_fails_safe(self):
        with patch('port_resolution_backend.run_feature_residual_review',return_value=base()),patch('port_resolution_backend.sha',side_effect=RuntimeError('unexpected read')):
            self.assertEqual(run_resolution_review({},resolution_enabled=True)['resolution_policy']['fallback_reason'],'accepted_feature_branch_not_available')
if __name__=='__main__':unittest.main()
