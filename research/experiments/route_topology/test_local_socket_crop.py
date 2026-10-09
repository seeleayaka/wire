import unittest
import numpy as np
from local_socket_crop import context_box,native_coverage


class LocalContextTests(unittest.TestCase):
    def test_identity_and_translation(self):
        self.assertEqual(context_box([100,100,200,150],np.eye(3),[500,500]),[0,0,300,250])
        m=np.array([[1,0,-20],[0,1,-30],[0,0,1]])
        self.assertEqual(context_box([100,100,200,150],m,[500,500]),[20,30,320,280])

    def test_bounds_rejected(self):
        with self.assertRaises(ValueError):context_box([0,0,20,10],np.eye(3),[100,100])

    def test_context_not_exact_and_no_pixel_mutation(self):
        rgb=np.zeros((20,20,3),np.uint8);rgb[:10,:10]=[255,0,0];rgb[10:,:10]=[0,128,255]
        raw=np.ones((20,20),bool);region=np.zeros_like(raw);region[:,10:]=True
        saved=rgb.copy(); coverage=native_coverage(rgb,region,raw,.95)
        self.assertFalse(coverage['local_two_color_coverage_candidate'])
        self.assertEqual(coverage['context_color_hits'],[100,100]);np.testing.assert_array_equal(rgb,saved)

    def test_same_mask_score_required(self):
        rgb=np.zeros((20,20,3),np.uint8);rgb[:10]=[255,0,0];rgb[10:]=[0,128,255]
        region=np.ones((20,20),bool)
        self.assertFalse(native_coverage(rgb,region,region,.74)['local_two_color_coverage_candidate'])
        self.assertTrue(native_coverage(rgb,region,region,.75)['local_two_color_coverage_candidate'])
        top=region.copy();top[10:]=False
        self.assertFalse(native_coverage(rgb,region,top,.95)['local_two_color_coverage_candidate'])


if __name__=='__main__':unittest.main()
