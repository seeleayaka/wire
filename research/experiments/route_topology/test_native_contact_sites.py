import unittest
import numpy as np
from audit_typed_affine_contact_sites import native_sites

class NativeSiteTests(unittest.TestCase):
    def test_source_pixels_not_interpolated_evidence(self):
        rgb=np.zeros((6,6,3),np.uint8)
        self.assertEqual(native_sites(rgb,np.eye(3),np.ones((6,6),bool),(0,0,6,6)),[])

    def test_original_coordinate_translation(self):
        rgb=np.full((8,8,3),200,np.uint8)
        m=np.array([[1,0,-2],[0,1,-3],[0,0,1]],dtype=float)
        self.assertEqual(native_sites(rgb,m,np.ones((2,2),bool),(2,3,4,5)),[(0,0),(0,1),(1,0),(1,1)])

    def test_outside_crop_rejected(self):
        with self.assertRaises(ValueError):
            native_sites(np.full((6,6,3),200,np.uint8),np.eye(3),np.ones((6,6),bool),(2,2,4,4))

    def test_resampling_deduplicates_native_points(self):
        points=native_sites(np.full((6,6,3),200,np.uint8),np.diag([2.,2.,1.]),np.ones((4,4),bool),(0,0,6,6))
        self.assertEqual(len(points),len(set(points)))
        self.assertLess(len(points),16)

if __name__=='__main__':unittest.main()
