import unittest
from compound_socket_observer import resolve

class FrozenResolution(unittest.TestCase):
    def test_all_old_labels_preserved_despite_new_conflict(self):
        for old in [0,1]:
            for color in [None,0,1]:
                for semantic in [None,0,1]:
                    for positive in [False,True]:self.assertEqual(resolve(old,color,semantic,positive),old)
    def test_agreement_without_positive_contact_abstains(self):
        self.assertIsNone(resolve(None,0,0,False))
    def test_positive_contact_without_both_heads_abstains(self):
        for a,b in [(None,0),(0,None),(1,0),(0,1),(1,1)]:self.assertIsNone(resolve(None,a,b,True))
    def test_exposed_rescue_only(self):self.assertEqual(resolve(None,0,0,True),0)
    def test_reject_untyped_flags(self):
        with self.assertRaises(ValueError):resolve(None,0,0,1)

if __name__=='__main__':unittest.main()
