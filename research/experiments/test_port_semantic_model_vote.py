import unittest
from test_teacher_student_port_policy import fixtures
from port_semantic_model_vote import proposals
class SemanticModelVoteTests(unittest.TestCase):
    def test_two_distinct_checkpoints_required(self):
        teacher,student=fixtures();teacher['weight_sha256']='teacher';student['weight_sha256']='student'
        self.assertTrue(proposals(teacher,[teacher,student]))
        self.assertEqual(proposals(teacher,[student,student]),[])
    def test_two_resolutions_of_same_weight_are_not_two_votes(self):
        teacher,student=fixtures();teacher['weight_sha256']=student['weight_sha256']='same'
        self.assertEqual(proposals(teacher,[teacher,student]),[])
    def test_no_implicit_original_teacher_confidence_gate(self):
        teacher,student=fixtures();teacher['weight_sha256']='a';student['weight_sha256']='b'
        for r in teacher['predictions']['merged_predictions']:r['confidence']=.06
        for r in student['predictions']['merged_predictions']:r['confidence']=.06
        self.assertTrue(proposals(teacher,[teacher,student]))
    def test_boundary_floor_is_not_accepted(self):
        teacher,student=fixtures();teacher['weight_sha256']='a';student['weight_sha256']='b'
        for r in student['predictions']['merged_predictions']:r['confidence']=.05
        self.assertEqual(proposals(teacher,[teacher,student]),[])
    def test_changed_source_rejected(self):
        teacher,student=fixtures();student['source_sha256']='changed'
        with self.assertRaises(ValueError):proposals(teacher,[teacher,student])
if __name__=='__main__':unittest.main()
