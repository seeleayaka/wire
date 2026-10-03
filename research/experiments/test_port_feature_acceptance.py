import unittest
from evaluate_port_feature_adaptation import qualifies
class AcceptanceTests(unittest.TestCase):
    def test_inner_gain(self):self.assertTrue(qualifies('inner',dict(tp=61,unmatched=0),dict(tp=62,unmatched=0),0,0))
    def test_inner_flat_rejected(self):self.assertFalse(qualifies('inner',dict(tp=61,unmatched=0),dict(tp=61,unmatched=0),0,0))
    def test_outer_flat_allowed(self):self.assertTrue(qualifies('outer',dict(tp=31,unmatched=1),dict(tp=31,unmatched=1),0,0))
    def test_lost_target_rejected_despite_total_gain(self):self.assertFalse(qualifies('train',dict(tp=67,unmatched=0),dict(tp=70,unmatched=0),0,1))
    def test_unmatched_increase_rejected(self):self.assertFalse(qualifies('train',dict(tp=67,unmatched=0),dict(tp=70,unmatched=1),0,0))
    def test_normal_cues_rejected(self):self.assertFalse(qualifies('train',dict(tp=67,unmatched=0),dict(tp=70,unmatched=0),1,0))
if __name__=='__main__':unittest.main()
