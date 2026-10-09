import unittest
import numpy as np
import cv2
from socket_edge_caliper import compare

class CaliperTests(unittest.TestCase):
    def fixture(self):
        x = np.zeros((50,100,3),np.uint8)
        cv2.rectangle(x,(20,12),(78,35),(180,180,180),2)
        cv2.line(x,(30,15),(60,32),(120,120,120),2)
        return x
    def test_identical_geometry_is_not_connection_verdict(self):
        x=self.fixture(); r=compare(x,x)
        self.assertTrue(r['reference_geometry_compatible'])
        self.assertEqual(r['decision'],'insufficient_evidence')
        self.assertEqual(r['new_confirmed_connections'],0)
    def test_blank_is_unknown(self):
        self.assertFalse(compare(self.fixture(),np.zeros((50,100,3),np.uint8))['reference_geometry_compatible'])
    def test_small_pose_tolerance(self):
        x=self.fixture()
        self.assertTrue(compare(x,np.roll(x,(1,-1),(0,1)))['reference_geometry_compatible'])
    def test_wrong_geometry_rejects(self):
        x=self.fixture(); y=np.zeros_like(x)
        cv2.circle(y,(50,25),15,(255,255,255),2)
        self.assertFalse(compare(x,y)['reference_geometry_compatible'])
    def test_wrong_size_rejects(self):
        with self.assertRaises(ValueError): compare(self.fixture(),np.zeros((5,10,3),np.uint8))

if __name__=='__main__': unittest.main()
