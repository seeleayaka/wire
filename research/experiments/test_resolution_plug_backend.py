import copy,unittest
from unittest.mock import patch
import numpy as np
from test_port_residual_feature_backend import base,cue
from port_resolution_plug_backend import append_plugs,run_plug_review,POLICY_ID,WEIGHT_SHA
class PlugBackendTests(unittest.TestCase):
    def test_append_provenance_preserves_original(self):
        old=base();before=copy.deepcopy(old);output,added=append_plugs(old,[cue()],[],np.eye(3))
        self.assertEqual(old,before);self.assertEqual(len(added),1)
        self.assertEqual(output['rescue_hints'],old['rescue_hints'])
        self.assertEqual(added[0]['resolution_policy_id'],POLICY_ID)
        self.assertEqual(added[0]['student_weight_sha256'],WEIGHT_SHA)
        self.assertEqual(added[0]['inference_imgsz'],1280)
        self.assertTrue(added[0]['loose_plug_only']);self.assertFalse(added[0]['automatic_fault_verdict'])
    def test_new_empty_jack_rejected(self):
        row=cue();row['class_id']=1
        with self.assertRaises(ValueError):append_plugs(base(),[row],[],np.eye(3))
    def test_reference_veto(self):
        ref=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_plugs(base(),[cue()],[ref],np.eye(3))[1],[])
    def test_roi_and_warp_gates(self):
        old=base();old['analysis_rois']=[dict(left=500,top=500,right=900,bottom=900)]
        self.assertEqual(append_plugs(old,[cue()],[],np.eye(3))[1],[])
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_plugs(base(),[cue()],[],matrix)[1],[])
    def test_existing_supplement_budget(self):
        old=base();old['supplementary_hints']=[dict(box=dict(left=700+i*80,top=900,right=750+i*80,bottom=950)) for i in range(5)]
        output,added=append_plugs(old,[cue()],[],np.eye(3))
        self.assertEqual(added,[]);self.assertEqual(output['supplementary_hints'],old['supplementary_hints'])
    def test_disabled_no_new_reads(self):
        with patch('port_resolution_plug_backend.run_feature_residual_review',return_value=base()),patch('port_resolution_plug_backend.sha',side_effect=RuntimeError('unexpected new read')):
            self.assertFalse(run_plug_review({})['resolution_policy']['enabled'])
    def test_unavailable_accepted_branch_falls_back(self):
        with patch('port_resolution_plug_backend.run_feature_residual_review',return_value=base()),patch('port_resolution_plug_backend.sha',side_effect=RuntimeError('unexpected new read')):
            self.assertEqual(run_plug_review({},resolution_enabled=True)['resolution_policy']['fallback_reason'],'accepted_feature_branch_not_available')
if __name__=='__main__':unittest.main()
