import copy,unittest
from unittest.mock import patch
import numpy as np
from teacher_student_port_backend import append_verified_student,run_teacher_student_review

def base():return dict(status='applied',rescue_hints=[],supplementary_hints=[],existing_hints=[],
    source_evidence=dict(predictions=dict(source_shape=[2736,3648])),
    reference_evidence=dict(predictions=dict(source_shape=[2736,3648]),aligned_predictions=[]),
    analysis_rois=[dict(left=0,top=0,right=3648,bottom=2736)])
def candidate(x=100):return dict(box_xyxy=[x,100,x+40,140],class_id=0,confidence=.9)

class BackendTests(unittest.TestCase):
    def test_add_without_mutating_old(self):
        old=base();snapshot=copy.deepcopy(old);result,added=append_verified_student(old,[candidate()],[],np.eye(3))
        self.assertEqual(len(added),1);self.assertEqual(old,snapshot);self.assertEqual(result['rescue_hints'],old['rescue_hints'])
    def test_reference_static_veto(self):
        ref=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_verified_student(base(),[candidate()],[ref],np.eye(3))[1],[])
    def test_analysis_roi_respected(self):
        old=base();old['analysis_rois']=[dict(left=500,top=500,right=1000,bottom=1000)]
        self.assertEqual(append_verified_student(old,[candidate()],[],np.eye(3))[1],[])
    def test_invalid_warp_coverage_rejected(self):
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_verified_student(base(),[candidate()],[],matrix)[1],[])
    def test_supplementary_budget(self):
        result,added=append_verified_student(base(),[candidate(100+i*80) for i in range(8)],[],np.eye(3))
        self.assertEqual(len(added),5);self.assertEqual(len(result['supplementary_hints']),5)
    def test_existing_cue_preserved_and_deduped(self):
        old=base();old['rescue_hints']=[dict(box=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.95))]
        result,added=append_verified_student(old,[candidate()],[],np.eye(3))
        self.assertEqual(added,[]);self.assertEqual(result['rescue_hints'],old['rescue_hints'])
    def test_disabled_does_not_access_weights(self):
        with patch('teacher_student_port_backend.run_context_port_recheck',return_value=base()),patch('teacher_student_port_backend.sha',side_effect=RuntimeError('must not read')):
            result=run_teacher_student_review({})
            self.assertFalse(result['teacher_student_policy']['enabled'])
    def test_failed_student_keeps_teacher(self):
        old=base();old['rescue_hints']=[dict(box=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.95))]
        with patch('teacher_student_port_backend.run_context_port_recheck',return_value=copy.deepcopy(old)),patch('teacher_student_port_backend.sha',side_effect=ValueError('identity_fail')):
            result=run_teacher_student_review({'inspection':'fake','reference':'fake'},enabled=True,supplementary_enabled=True,student_enabled=True)
            self.assertEqual(result['rescue_hints'],old['rescue_hints']);self.assertIn('identity_fail',result['teacher_student_policy']['fallback_reason'])

if __name__=='__main__':unittest.main()
