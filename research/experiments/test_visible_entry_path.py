import unittest
import json
import numpy as np
from visible_entry_path import inspect_local_path


class PathTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.zeros((40,40,3),dtype=np.uint8)
        self.mask=np.zeros((40,40),dtype=bool)
        self.box=[10,5,14,10]

    def test_direct_mask(self):
        self.mask[8:25,11]=True
        r=inspect_local_path(self.rgb,self.mask,self.box)
        self.assertTrue(r['continuous_pixel_support'])
        self.assertEqual(r['added_blue_path_pixels'],0)
        self.assertFalse(r['confirmed_assignment'])
        json.dumps(r)

    def test_blue_bridge(self):
        self.rgb[8:20,11]=[0,0,180];self.mask[20:25,11]=True
        r=inspect_local_path(self.rgb,self.mask,self.box)
        self.assertTrue(r['continuous_pixel_support'])
        self.assertGreater(r['added_blue_path_pixels'],0)

    def test_gap_not_filled(self):
        self.rgb[8:19,11]=[0,0,180];self.mask[20:25,11]=True
        self.assertFalse(inspect_local_path(self.rgb,self.mask,self.box)['continuous_pixel_support'])

    def test_adjacent_not_joined(self):
        self.rgb[8:20,11]=[0,0,180];self.mask[20:25,12]=True
        self.assertFalse(inspect_local_path(self.rgb,self.mask,self.box)['continuous_pixel_support'])

    def test_nonblue_no_bridge(self):
        self.rgb[8:20,11]=[180,0,0];self.mask[20:25,11]=True
        self.assertFalse(inspect_local_path(self.rgb,self.mask,self.box)['continuous_pixel_support'])

    def test_touching_same_color_is_not_identity_proof(self):
        self.rgb[8:20,11]=[0,0,180]
        self.rgb[18,11:16]=[0,0,180]
        self.mask[18:25,15]=True
        r=inspect_local_path(self.rgb,self.mask,self.box)
        self.assertTrue(r['continuous_pixel_support'])
        self.assertFalse(r['confirmed_assignment'])

    def test_empty_and_validation(self):
        self.assertFalse(inspect_local_path(self.rgb,self.mask,self.box)['continuous_pixel_support'])
        with self.assertRaises(ValueError):
            inspect_local_path(self.rgb,self.mask,[10,5,41,10])

if __name__=='__main__':
    unittest.main()
