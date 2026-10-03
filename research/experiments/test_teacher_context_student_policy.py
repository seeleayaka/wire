import copy,unittest
from unittest.mock import patch
from teacher_context_student_policy import confirm_entries,extend
def row(score=.9,cls=0,box=None):return dict(class_id=cls,confidence=score,box_xyxy=box or [100,100,140,140])
def entry():return dict(proposal=row(),windows=[[0,0,1280,1280],[80,80,1360,1360]],views=[[row()],[row()]])
class ConfirmationTests(unittest.TestCase):
    def test_both_views_required(self):
        e=entry();e['views'][1]=[];self.assertEqual(confirm_entries([e]),[])
    def test_same_class_required(self):
        e=entry();e['views'][1]=[row(cls=1)];self.assertEqual(confirm_entries([e]),[])
    def test_each_score_strict(self):
        e=entry();e['views'][1]=[row(score=.75)];self.assertEqual(confirm_entries([e]),[])
    def test_each_box_matches_seed(self):
        e=entry();e['views'][1]=[row(box=[150,100,190,140])];self.assertEqual(confirm_entries([e]),[])
    def test_distinct_windows_required(self):
        e=entry();e['windows'][1]=e['windows'][0];self.assertEqual(confirm_entries([e]),[])
    def test_seed_preserved(self):
        e=entry();before=copy.deepcopy(e);p=confirm_entries([e])[0]
        self.assertEqual(p['box_xyxy'],e['proposal']['box_xyxy']);self.assertEqual(e,before)
    def test_full_budget_preserved(self):
        old=dict(primary=[row()]*5,all_predictions=[row()]*10,student_additions=[])
        with patch('teacher_context_student_policy.merge',return_value=copy.deepcopy(old)),patch('teacher_context_student_policy.proposals',return_value=[row()]):
            self.assertEqual(extend({}, {}, [entry()])['teacher_context_additions'],[])
    def test_unproposed_entry_rejected(self):
        old=dict(primary=[],all_predictions=[],student_additions=[])
        with patch('teacher_context_student_policy.merge',return_value=copy.deepcopy(old)),patch('teacher_context_student_policy.proposals',return_value=[]):
            self.assertEqual(extend({}, {}, [entry()])['all_predictions'],[])
if __name__=='__main__':unittest.main()
