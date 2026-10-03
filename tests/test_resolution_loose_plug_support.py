import copy,unittest
from unittest.mock import patch
import numpy as np
from test_feature_residual_port_support import row,case,base
from inspection_agent.teacher_student_port_support import complementary_candidates
from inspection_agent.resolution_loose_plug_support import resolution_candidates,append_plugs,run_resolution_plug_review,POLICY_ID
def fixture():
    teacher=case([row(100+i*60,.99-i*.005) for i in range(5)]+[row(1500,.3)])
    new=case([row(1500)]);new['predictions']['edge_kept_predictions'].append(row(1500,.85,1))
    current=complementary_candidates(teacher,case([]))
    return teacher,current,new
class ResolutionPlugTests(unittest.TestCase):
    def test_source_append_preserves_inputs_and_old_prefix(self):
        teacher,current,new=fixture();before=copy.deepcopy((teacher,current,new))
        output=resolution_candidates(teacher,current,new)
        self.assertEqual(len(output['resolution_additions']),1)
        self.assertEqual(output['all_predictions'][:len(current['all_predictions'])],current['all_predictions'])
        self.assertEqual((teacher,current,new),before)
    def test_empty_jack_not_added(self):
        teacher,current,new=fixture()
        for m in (teacher,new):
            for key in ('merged_predictions','edge_kept_predictions'):
                for r in m['predictions'][key]:r['class_id']=1
        self.assertEqual(resolution_candidates(teacher,current,new)['resolution_additions'],[])
    def test_identity_and_geometry_fail_safe(self):
        for kind in ('source','shape'):
            teacher,current,new=fixture()
            if kind=='source':new['source_sha256']='changed'
            else:new['predictions']['source_shape']=[1,1]
            output=resolution_candidates(teacher,current,new)
            self.assertEqual(output['all_predictions'],current['all_predictions'])
            self.assertIsNotNone(output['resolution_fallback_reason'])
    def test_shared_budget(self):
        teacher,current,new=fixture();current['all_predictions']+=[row(700+i*60) for i in range(5)]
        self.assertEqual(resolution_candidates(teacher,current,new)['resolution_additions'],[])
    def test_last_shared_slot_skips_disallowed_jack_first(self):
        teacher,current,new=fixture();current['all_predictions']+=[row(700+i*60) for i in range(4)]
        jack=row(1800,.99);jack['class_id']=1;plug=row(1500,.8);old=copy.deepcopy(current)
        with patch('inspection_agent.resolution_loose_plug_support.complementary_candidates',
                   return_value=dict(fallback_reason=None,student_additions=[jack,plug])):
            output=resolution_candidates(teacher,current,new)
        self.assertEqual(len(output['resolution_additions']),1)
        self.assertEqual(output['resolution_additions'][0]['class_id'],0)
        self.assertEqual(len(output['all_predictions']),10)
        self.assertEqual(output['all_predictions'][:9],old['all_predictions'])
        self.assertEqual(current,old)
    def test_last_shared_slot_skips_duplicate_before_distinct_plug(self):
        teacher,current,new=fixture();current['all_predictions']+=[row(700+i*60) for i in range(4)]
        old=copy.deepcopy(current);duplicate=copy.deepcopy(current['all_predictions'][0]);plug=row(1500,.8)
        with patch('inspection_agent.resolution_loose_plug_support.complementary_candidates',
                   return_value=dict(fallback_reason=None,student_additions=[duplicate,plug])):
            output=resolution_candidates(teacher,current,new)
        self.assertEqual(len(output['resolution_additions']),1)
        self.assertEqual(output['resolution_additions'][0]['box_xyxy'],plug['box_xyxy'])
        self.assertEqual(output['all_predictions'][:9],old['all_predictions'])
        self.assertEqual(current,old)
    def test_metadata_and_reference_veto(self):
        cue=row(100);ref=dict(left=100,top=300,right=140,bottom=340,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_plugs(base(),[cue],[ref],np.eye(3))[1],[])
        _,added=append_plugs(base(),[cue],[],np.eye(3))
        self.assertEqual(added[0]['resolution_policy_id'],POLICY_ID)
        self.assertTrue(added[0]['loose_plug_only']);self.assertFalse(added[0]['automatic_fault_verdict'])
    def test_disallowed_native_candidate_rejected(self):
        cue=row(100);cue['class_id']=1
        with self.assertRaises(ValueError):append_plugs(base(),[cue],[],np.eye(3))
    def test_roi_and_invalid_warp(self):
        old=base();old['analysis_rois']=[dict(left=500,top=500,right=900,bottom=900)]
        self.assertEqual(append_plugs(old,[row(100)],[],np.eye(3))[1],[])
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_plugs(base(),[row(100)],[],matrix)[1],[])
    def test_disabled_no_fingerprint_reads(self):
        with patch('inspection_agent.resolution_loose_plug_support.run_feature_residual_review',return_value=base()),patch(
            'inspection_agent.resolution_loose_plug_support.resolution_runtime_fingerprint',side_effect=RuntimeError('no new reads')):
            self.assertFalse(run_resolution_plug_review({},project='unused')['resolution_policy']['enabled'])
    def test_unavailable_accepted_branch_preserved(self):
        with patch('inspection_agent.resolution_loose_plug_support.run_feature_residual_review',return_value=base()):
            self.assertEqual(run_resolution_plug_review({},project='unused',resolution_enabled=True)['resolution_policy']['fallback_reason'],
                'accepted_feature_branch_not_available')
    def test_bad_manifest_preserves_original(self):
        old=base();old['feature_residual_evidence']={};before=copy.deepcopy(old)
        with patch('inspection_agent.resolution_loose_plug_support.run_feature_residual_review',return_value=old),patch(
            'inspection_agent.resolution_loose_plug_support.resolution_runtime_fingerprint',return_value={'resolution_manifest':'changed'}):
            result=run_resolution_plug_review({},project='unused',resolution_enabled=True)
        self.assertIn('resolution_provenance_manifest_mismatch',result['resolution_policy']['fallback_reason'])
        self.assertEqual(result['rescue_hints'],before['rescue_hints']);self.assertEqual(result['supplementary_hints'],before['supplementary_hints'])
if __name__=='__main__':unittest.main()
