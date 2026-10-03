import unittest
import numpy as np
from directional_entry_path import inspect_directional_path
from visible_entry_path import inspect_local_path


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.zeros((50,50,3),dtype=np.uint8);self.mask=np.zeros((50,50),dtype=bool)
        self.box=[10,5,14,10]

    def run_path(self, axis=(0,1)):
        return inspect_directional_path(self.rgb,self.mask,self.box,axis)

    def test_straight_retained(self):
        self.rgb[8:20,11]=[0,0,180];self.mask[20:30,11]=True
        self.assertTrue(self.run_path()['continuous_pixel_support'])

    def test_horizontal_shortcut_rejected(self):
        self.rgb[8,11:20]=[0,0,180];self.mask[8:20,20]=True
        self.assertTrue(inspect_local_path(self.rgb,self.mask,self.box)['continuous_pixel_support'])
        self.assertFalse(self.run_path()['continuous_pixel_support'])

    def test_moderate_bend_retained(self):
        self.rgb[8:20,11]=[0,0,180];self.rgb[19,11:16]=[0,0,180]
        self.rgb[19:25,15]=[0,0,180];self.mask[25:30,15]=True
        self.assertTrue(self.run_path()['continuous_pixel_support'])

    def test_sharp_real_bend_can_be_rejected(self):
        self.rgb[8,11:20]=[0,0,180];self.mask[8:20,20]=True
        # A genuine 90-degree turn is indistinguishable here from a shortcut.
        self.assertFalse(self.run_path()['continuous_pixel_support'])

    def test_gap_not_filled(self):
        self.rgb[8:19,11]=[0,0,180];self.mask[20:30,11]=True
        self.assertFalse(self.run_path()['continuous_pixel_support'])

    def test_no_backward(self):
        self.rgb[2:9,11]=[0,0,180];self.mask[2,11]=True
        self.assertFalse(self.run_path()['continuous_pixel_support'])

    def test_explicit_axis_validation(self):
        for axis in (None,(1,1),(0,True)):
            with self.assertRaises(ValueError):self.run_path(axis)

    def test_rotation_equivariance(self):
        self.rgb[8:20,11]=[0,0,180];self.mask[20:30,11]=True
        rotated_rgb=np.rot90(self.rgb);rotated_mask=np.rot90(self.mask)
        # np.rot90 maps source(x,y) to (y,49-x), so downward becomes rightward.
        rotated_box=[5,36,10,40]
        result=inspect_directional_path(rotated_rgb,rotated_mask,rotated_box,[1,0])
        self.assertEqual(result['continuous_pixel_support'],self.run_path()['continuous_pixel_support'])

if __name__=='__main__':unittest.main()
