import unittest
from evaluate_port_multiscale_candidate import acceptance,load,BASE

class AcceptanceTests(unittest.TestCase):
    def setUp(self):self.protocol=load(BASE/'evaluation_protocol.json')
    def inner(self,tp=58,unmatched=0,normal=0,targets=80):
        return acceptance('inner',dict(tp=tp,unmatched=unmatched,targets=targets),normal,self.protocol)['source_only_pass']
    def test_requires_real_gain(self):self.assertFalse(self.inner(tp=57));self.assertTrue(self.inner())
    def test_more_false_cues_rejected(self):self.assertFalse(self.inner(unmatched=1))
    def test_normal_cue_rejected(self):self.assertFalse(self.inner(normal=1))
    def test_missing_targets_rejected(self):self.assertFalse(self.inner(targets=79))
    def test_outer_no_regression(self):
        def check(tp,unmatched):return acceptance('outer',dict(tp=tp,unmatched=unmatched,targets=56),0,self.protocol)['source_only_pass']
        self.assertTrue(check(31,1));self.assertFalse(check(30,1));self.assertFalse(check(32,2))
    def test_train_is_never_acceptance(self):
        self.assertFalse(acceptance('train',{},0,self.protocol)['source_only_pass'])

if __name__=='__main__':unittest.main()
