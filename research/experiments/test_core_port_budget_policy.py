import unittest,copy
from core_port_budget_policy import bounded_expand
def p(score,l=0):return dict(confidence=score,class_id=0,box_xyxy=[l,0,l+10,10])
class BudgetTests(unittest.TestCase):
    def test_original_top1_kept(self):self.assertEqual(bounded_expand([p(.3)]),[p(.3)])
    def test_add_only_high_scores(self):
        self.assertEqual([r['confidence'] for r in bounded_expand([p(.9),p(.6,20),p(.5,40),p(.4,60)])],[.9,.6])
    def test_maximum_is_five(self):self.assertEqual(len(bounded_expand([p(.9,i*20) for i in range(9)])),5)
    def test_empty(self):self.assertEqual(bounded_expand([]),[])
    def test_disabled_by_no_strong_base(self):self.assertEqual(bounded_expand([p(.25)]),[])
    def test_inputs_unchanged(self):
        rows=[p(.6),p(.8,20)];old=copy.deepcopy(rows);bounded_expand(rows);self.assertEqual(rows,old)
    def test_invalid_policy(self):
        with self.assertRaises(ValueError):bounded_expand([],maximum=6)
    def test_budget_one_equivalent(self):self.assertEqual(bounded_expand([p(.9),p(.8,20)],maximum=1),[p(.9)])
if __name__=='__main__':unittest.main()
