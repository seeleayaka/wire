import unittest
from component_rescue import rescue
def p(k):return dict(visual_label_candidate=k,decision='insufficient_evidence')
class RescueTests(unittest.TestCase):
    def test_old_singletons_never_overridden(self):
        for old in [0,1]:
            r=rescue(p(old),p(1-old),p(1-old),True,True)
            self.assertEqual(r['visual_label_candidate'],old);self.assertFalse(r['component_rescue_applied'])
    def test_missing_gate_and_pose_abstain(self):
        for source,pose in [(False,True),(True,False),(False,False)]:
            self.assertIsNone(rescue(p(None),p(1),p(1),source,pose)['visual_label_candidate'])
    def test_feature_disagreement_abstains(self):
        for c,s in [(0,1),(None,1),(0,None),(None,None)]:
            self.assertIsNone(rescue(p(None),p(c),p(s),True,True)['visual_label_candidate'])
    def test_same_feature_views_still_one_observer_not_connection(self):
        r=rescue(p(None),p(1),p(1),True,True)
        self.assertEqual(r['visual_label_candidate'],1);self.assertEqual(r['independent_observer_count'],1)
        self.assertEqual(r['decision'],'insufficient_evidence');self.assertEqual(r['new_confirmed_connections'],0)
if __name__=='__main__':unittest.main()
