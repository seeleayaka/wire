import unittest
import numpy as np
import cv2
from quarter_turn_view import forward,restore,prompt


class RotationTests(unittest.TestCase):
    def test_non_square_exact_rgb_roundtrip_and_no_mutation(self):
        rgb=np.arange(5*8*3,dtype=np.uint8).reshape(5,8,3);before=rgb.copy()
        view=forward(rgb)
        np.testing.assert_array_equal(np.rot90(view,-1),rgb)
        np.testing.assert_array_equal(rgb,before)

    def test_mask_components_pixels_and_boundary_preserved(self):
        raw=np.zeros((11,17),np.uint8);raw[2:5,2:5]=1;raw[8,15]=1
        view=np.rot90(raw,1); recovered=restore(view,[17,11])
        np.testing.assert_array_equal(raw,recovered)
        self.assertEqual(cv2.connectedComponents(raw,connectivity=8)[0],cv2.connectedComponents(view.copy(),connectivity=8)[0])
        self.assertEqual(int(raw.sum()),int(view.sum()))

    def test_box_rotation_matches_half_open_pixel_edges(self):
        # x=[2,6), y=[1,4) in width10/height5 ->x'=[1,4), y'=[4,8).
        self.assertEqual(prompt([.4,.5,.4,.6]),[.5,.6,.6,.4])
        raw=np.zeros((5,10),np.uint8);raw[1:4,2:6]=1
        np.testing.assert_array_equal(np.rot90(raw,1)[4:8,1:4],np.ones((4,3),np.uint8))

    def test_invalid_frame_and_box_fail_closed(self):
        with self.assertRaises(ValueError):restore(np.zeros((5,10)),[10,5])
        with self.assertRaises(ValueError):prompt([.9,.5,.4,.2])


if __name__=='__main__':unittest.main()
