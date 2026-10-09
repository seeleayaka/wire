import unittest
from semantic_visible_socket_observer import visible_rescue

class VisibleRescueTests(unittest.TestCase):
    def test_old_candidates_preserved(self):
        for old in [0,1]:self.assertEqual(visible_rescue(old,[None]*9),old)
    def test_only_all_nine_visible_can_rescue(self):
        self.assertEqual(visible_rescue(None,[1]*9),1)
        for last in [None,0]:self.assertIsNone(visible_rescue(None,[1]*8+[last]))
    def test_exposed_never_rescued(self):self.assertIsNone(visible_rescue(None,[0]*9))
    def test_wrong_count_or_untyped_rejected(self):
        for labels in [[1]*8,[True]*9,[2]*9]:
            with self.assertRaises(ValueError):visible_rescue(None,labels)
    def test_input_not_mutated(self):
        labels=[1]*9;visible_rescue(None,labels);self.assertEqual(labels,[1]*9)
if __name__=='__main__':unittest.main()
