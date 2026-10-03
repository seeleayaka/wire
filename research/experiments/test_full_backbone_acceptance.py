import copy,unittest
from test_teacher_student_port_policy import fixtures,case
from teacher_student_port_policy import merge
from evaluate_full_backbone_acceptance import append_backbone
class BackboneAcceptanceTests(unittest.TestCase):
    def test_append_and_preserve(self):
        teacher,new=fixtures();current=merge(teacher,case([]));before=copy.deepcopy((teacher,current,new))
        output=append_backbone(teacher,current,new)
        self.assertEqual(len(output['backbone_additions']),1)
        self.assertEqual(output['all_predictions'][:len(current['all_predictions'])],current['all_predictions'])
        self.assertEqual((teacher,current,new),before)
    def test_shared_budget(self):
        teacher,new=fixtures(True);current=merge(teacher,case([]))
        self.assertEqual(append_backbone(teacher,current,new)['backbone_additions'],[])
    def test_identity_fail_safe(self):
        teacher,new=fixtures();current=merge(teacher,case([]));new['source_sha256']='changed'
        result=append_backbone(teacher,current,new)
        self.assertIsNotNone(result['backbone_fallback_reason'])
        self.assertEqual(result['all_predictions'],current['all_predictions'])
    def test_duplicate_preserved(self):
        teacher,new=fixtures();current=merge(teacher,new)
        self.assertEqual(append_backbone(teacher,current,new)['backbone_additions'],[])
if __name__=='__main__':unittest.main()
