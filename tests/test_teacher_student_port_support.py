import copy,unittest
from unittest.mock import patch
import numpy as np
from inspection_agent.teacher_student_port_support import append_verified_student,run_teacher_student_review,valid

def base():return dict(status='applied',rescue_hints=[],supplementary_hints=[],existing_hints=[],
    source_evidence=dict(predictions=dict(source_shape=[2736,3648])),
    reference_evidence=dict(predictions=dict(source_shape=[2736,3648]),aligned_predictions=[]),
    analysis_rois=[dict(left=0,top=0,right=3648,bottom=2736)])
def candidate(x=100):return dict(box_xyxy=[x,100,x+40,140],class_id=0,confidence=.9)
MODULE='inspection_agent.teacher_student_port_support'

class TeacherStudentSupportTests(unittest.TestCase):
    def test_add_without_mutating_old(self):
        old=base();before=copy.deepcopy(old);result,added=append_verified_student(old,[candidate()],[],np.eye(3))
        self.assertEqual(len(added),1);self.assertEqual(old,before);self.assertEqual(result['rescue_hints'],old['rescue_hints'])
    def test_reference_static_veto(self):
        ref=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.9,valid_warp_fraction=1)
        self.assertEqual(append_verified_student(base(),[candidate()],[ref],np.eye(3))[1],[])
    def test_analysis_roi_respected(self):
        old=base();old['analysis_rois']=[dict(left=500,top=500,right=1000,bottom=1000)]
        self.assertEqual(append_verified_student(old,[candidate()],[],np.eye(3))[1],[])
    def test_invalid_warp_rejected(self):
        matrix=np.eye(3);matrix[0,2]=4000
        self.assertEqual(append_verified_student(base(),[candidate()],[],matrix)[1],[])
    def test_budget_and_original_supplements(self):
        old=base();old['supplementary_hints']=[dict(box=dict(left=700,top=900,right=750,bottom=950))]
        result,added=append_verified_student(old,[candidate(100+i*80) for i in range(8)],[],np.eye(3))
        self.assertEqual(len(added),4);self.assertEqual(len(result['supplementary_hints']),5)
        self.assertEqual(result['supplementary_hints'][0],old['supplementary_hints'][0])
    def test_existing_cue_preserved_and_deduped(self):
        old=base();old['rescue_hints']=[dict(box=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.95))]
        result,added=append_verified_student(old,[candidate()],[],np.eye(3))
        self.assertEqual(added,[]);self.assertEqual(result['rescue_hints'],old['rescue_hints'])
    def test_disabled_does_not_read_student(self):
        with patch(MODULE+'.run_context_port_recheck',return_value=base()),patch(MODULE+'.sha',side_effect=RuntimeError('must not read')):
            result=run_teacher_student_review({},project='unused')
            self.assertFalse(result['teacher_student_policy']['enabled'])
    def test_bad_manifest_keeps_teacher(self):
        old=base();old['rescue_hints']=[dict(box=dict(left=100,top=100,right=140,bottom=140,class_id=0,confidence=.95))]
        with patch(MODULE+'.run_context_port_recheck',return_value=copy.deepcopy(old)),patch(MODULE+'.sha',return_value='wrong'):
            result=run_teacher_student_review({},project='unused',enabled=True,supplementary_enabled=True,student_enabled=True)
            self.assertEqual(result['rescue_hints'],old['rescue_hints'])
            self.assertIn('student_provenance_manifest_mismatch',result['teacher_student_policy']['fallback_reason'])
    def test_supplement_disabled_never_reads_student(self):
        with patch(MODULE+'.run_context_port_recheck',return_value=base()),patch(MODULE+'.sha',side_effect=RuntimeError('must not read')):
            result=run_teacher_student_review({},project='unused',enabled=True,student_enabled=True)
            self.assertEqual(result['teacher_student_policy']['added_hints'],0)
    def test_full_budget_never_reads_student(self):
        old=base();old['supplementary_hints']=[dict(box={}) for _ in range(5)]
        with patch(MODULE+'.run_context_port_recheck',return_value=old),patch(MODULE+'.sha',side_effect=RuntimeError('must not read')):
            result=run_teacher_student_review({},project='unused',enabled=True,supplementary_enabled=True,student_enabled=True)
            self.assertEqual(result['teacher_student_policy']['added_hints'],0)
    def test_invalid_geometry_and_scores(self):
        for changes in (dict(confidence=float('nan')),dict(confidence=1.1),dict(class_id=True),dict(box_xyxy=[10,10,9,9])):
            row=candidate();row.update(changes);self.assertFalse(valid(row))
        self.assertTrue(valid(candidate()))
if __name__=='__main__':unittest.main()
