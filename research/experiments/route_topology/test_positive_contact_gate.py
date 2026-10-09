import unittest
import numpy as np
from run_positive_contact_gate import bright_contacts

class ContactTests(unittest.TestCase):
    def test_dark_occlusion_is_not_contact(self):
        self.assertFalse(bright_contacts(np.zeros((50,100,3),np.uint8)).any())
    def test_colored_wire_is_not_contact(self):
        rgb=np.zeros((50,100,3),np.uint8);rgb[:,:,0]=255
        self.assertFalse(bright_contacts(rgb).any())
    def test_bright_neutral_is_cue_not_identity(self):
        self.assertTrue(bright_contacts(np.full((50,100,3),200,np.uint8)).all())
    def test_no_gap_filling(self):
        rgb=np.zeros((50,100,3),np.uint8);rgb[:,5]=255;rgb[:,8]=255
        mask=bright_contacts(rgb)
        self.assertFalse(mask[:,6:8].any())

if __name__=='__main__':unittest.main()
