import unittest
from evaluate_port_residual_feature_support import qualifies


class GateTests(unittest.TestCase):
    def current(self):return dict(tp=61,unmatched=0)
    def test_gain_without_loss(self):
        self.assertTrue(qualifies('inner',self.current(),dict(tp=62,unmatched=0),0,0))
    def test_total_gain_cannot_hide_loss(self):
        self.assertFalse(qualifies('inner',self.current(),dict(tp=63,unmatched=0),0,1))
    def test_unmatched_and_normal_gates(self):
        self.assertFalse(qualifies('inner',self.current(),dict(tp=62,unmatched=1),0,0))
        self.assertFalse(qualifies('inner',self.current(),dict(tp=62,unmatched=0),1,0))
    def test_training_inner_need_positive_gain(self):
        for mode in ('train','inner'):
            self.assertFalse(qualifies(mode,self.current(),self.current(),0,0))
    def test_outer_can_hold_baseline(self):
        self.assertTrue(qualifies('outer',self.current(),self.current(),0,0))
    def test_broader_control_cannot_add_annotation_burden(self):
        old=dict(tp=0,unmatched=4)
        self.assertTrue(qualifies('extended',old,old,0,0))
        self.assertFalse(qualifies('extended',old,dict(tp=0,unmatched=5),0,0))


if __name__=='__main__':unittest.main()
