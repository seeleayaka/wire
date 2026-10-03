import copy,unittest
from test_teacher_student_port_policy import fixtures,case
from teacher_student_port_policy import merge
from port_resolution_plug_policy import append_resolution_plugs
class PlugPolicyTests(unittest.TestCase):
    def test_loose_plug_addition_and_preservation(self):
        teacher,new=fixtures();old=merge(teacher,case([]));before=copy.deepcopy((teacher,old,new))
        output=append_resolution_plugs(teacher,old,new)
        self.assertEqual(len(output['resolution_additions']),1)
        self.assertTrue(output['resolution_additions'][0]['loose_plug_only'])
        self.assertFalse(output['resolution_additions'][0]['automatic_fault_verdict'])
        self.assertEqual((teacher,old,new),before)
    def test_new_empty_jack_not_added(self):
        teacher,new=fixtures()
        for model in (teacher,new):
            for rows in model['predictions'].values():
                if isinstance(rows,list):
                    for row in rows:
                        if isinstance(row,dict):row['class_id']=1
        old=merge(teacher,case([]));output=append_resolution_plugs(teacher,old,new)
        self.assertEqual(output['resolution_additions'],[])
        self.assertEqual(output['all_predictions'],old['all_predictions'])
    def test_budget_and_identity_gates_inherited(self):
        teacher,new=fixtures(True);old=merge(teacher,case([]))
        self.assertEqual(append_resolution_plugs(teacher,old,new)['resolution_additions'],[])
        new['source_sha256']='changed'
        self.assertIsNotNone(append_resolution_plugs(teacher,old,new)['resolution_fallback_reason'])
if __name__=='__main__':unittest.main()
