import unittest
from scoped_affine_pose import choose_pose
class ScopedAffineTests(unittest.TestCase):
    def setUp(self):
        self.anchor=dict(id='A',kind='visible_lead_emergence')
        self.old=dict(id='A',localization_proposal_supported=False)
        self.new=dict(id='A',localization_proposal_supported=True,gates=dict(heldout=True,hull=True))
    def test_supported_old_is_identical(self):
        old=dict(self.old,localization_proposal_supported=True)
        self.assertIs(choose_pose(self.anchor,old,self.new),old)
    def test_emergence_can_use_gated_fallback(self):self.assertIs(choose_pose(self.anchor,self.old,self.new),self.new)
    def test_socket_cannot_use_fallback(self):self.assertIs(choose_pose(dict(self.anchor,kind='wire_entry_socket'),self.old,self.new),self.old)
    def test_failed_gate_not_overridden(self):
        new=dict(self.new,gates=dict(heldout=False))
        self.assertIs(choose_pose(self.anchor,self.old,new),self.old)
    def test_identity_mismatch_rejected(self):
        with self.assertRaises(ValueError):choose_pose(self.anchor,self.old,dict(self.new,id='B'))
if __name__=='__main__':unittest.main()
