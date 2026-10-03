import copy,unittest,sys
sys.path.insert(0,'E:/PythonProject10')
from unittest.mock import patch
from student_microcontext_policy import windows,extend
from test_teacher_context_student_policy import row,entry
class MicrocontextTests(unittest.TestCase):
    def test_scale_and_distinct_shifts(self):
        out=windows(row(box=[1700,1500,1740,1540]),[2736,3648]);self.assertNotEqual(out[0],out[1])
        self.assertTrue(all(p[2]-p[0]==640 and p[3]-p[1]==640 for p in out))
    def test_clipping_inside_image(self):
        for box in ([1,1,41,41],[3500,2650,3540,2690]):
            for l,t,r,b in windows(row(box=box),[2736,3648]):self.assertTrue(0<=l<r<=3648 and 0<=t<b<=2736)
    def test_small_image_rejected(self):
        with self.assertRaises(ValueError):windows(row(),[400,400])
    def test_existing_prefix_preserved(self):
        old=dict(primary=[row(box=[900,900,940,940])],all_predictions=[row(box=[900,900,940,940])],student_additions=[])
        with patch('student_microcontext_policy.merge',return_value=copy.deepcopy(old)),patch('student_microcontext_policy.proposals',return_value=[row()]):
            result=extend({}, {}, [entry()]);self.assertEqual(result['all_predictions'][0],old['all_predictions'][0]);self.assertEqual(len(result['microcontext_additions']),1)
            self.assertIn('student_microcontext_scores',result['microcontext_additions'][0]);self.assertNotIn('teacher_context_view_scores',result['microcontext_additions'][0])
    def test_budget(self):
        old=dict(primary=[row()]*5,all_predictions=[row()]*10,student_additions=[])
        with patch('student_microcontext_policy.merge',return_value=copy.deepcopy(old)),patch('student_microcontext_policy.proposals',return_value=[row()]):
            self.assertEqual(extend({}, {}, [entry()])['microcontext_additions'],[])
if __name__=='__main__':unittest.main()
