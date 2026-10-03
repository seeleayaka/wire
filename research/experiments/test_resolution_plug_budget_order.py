import copy,unittest
from test_feature_residual_port_support import row,case
from teacher_student_port_policy import merge
from port_resolution_plug_policy import append_resolution_plugs
from port_resolution_plug_policy_v2 import append_resolution_plugs_v2
class ClassBudgetOrderTests(unittest.TestCase):
    def test_disallowed_jack_does_not_spend_last_free_slot(self):
        teacher=case([row(100+i*60,.99-i*.005) for i in range(5)]+[row(1400,.3),row(1800,.3)])
        teacher['predictions']['merged_predictions'][-2]['class_id']=1
        jack=row(1400,.97);jack['class_id']=1;plug=row(1800,.92)
        alternative=case([jack,plug]);extra=copy.deepcopy([jack,plug])
        for entry in extra:entry.update(source_tile=1)
        alternative['predictions']['edge_kept_predictions']+=extra
        current=merge(teacher,case([]));current['all_predictions']+=[row(700+i*60,.9) for i in range(4)]
        before=copy.deepcopy((teacher,current,alternative))
        self.assertEqual(append_resolution_plugs(teacher,current,alternative)['resolution_additions'],[])
        output=append_resolution_plugs_v2(teacher,current,alternative)
        self.assertEqual(len(output['resolution_additions']),1)
        self.assertEqual(output['resolution_additions'][0]['class_id'],0)
        self.assertEqual(output['resolution_additions'][0]['box_xyxy'],plug['box_xyxy'])
        self.assertEqual(len(output['all_predictions']),10)
        self.assertEqual((teacher,current,alternative),before)
if __name__=='__main__':unittest.main()
