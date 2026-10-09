import unittest
import numpy as np
from crossing_pair_probe import pair_tangents

class PairTests(unittest.TestCase):
    def test_cross_pairs_straight_arms_not_everything_connected(self):
        r=pair_tangents([[1,0],[-1,0],[0,1],[0,-1]])
        self.assertEqual(r['pairs'],[(0,1),(2,3)])
        self.assertEqual(r['new_confirmed_connections'],0)
    def test_occluded_arm_not_fabricated(self):
        self.assertEqual(pair_tangents([[1,0],[-1,0],[0,1]])['status'],'insufficient_evidence')
    def test_coincident_directions_ambiguous(self):
        self.assertEqual(pair_tangents([[1,0],[1,0],[-1,0],[-1,0]])['status'],'insufficient_evidence')
    def test_rotation_scale_invariance(self):
        a=np.array([[1,0],[-1,0],[0,1],[0,-1]])
        for angle in np.linspace(0,6.28,31):
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            self.assertEqual(pair_tangents(a@rotation*3)['pairs'],[(0,1),(2,3)])
if __name__=='__main__': unittest.main()
