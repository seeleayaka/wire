import unittest
import numpy as np
from reference_color_paths import hue_families,observe_paths


class ColorPathTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((50,100,3),20,np.uint8);rgb[20:23,10:90]=[220,20,20];rgb[25:28,10:90]=[20,20,220]
        a=np.zeros((50,100),bool);b=a.copy();a[15:35,8:18]=True;b[15:35,82:92]=True
        return rgb,{'A':a,'B':b}

    def test_two_native_color_paths_not_electrical_identity(self):
        rgb,regions=self.fixture();result,_=observe_paths(rgb,regions,[[0,17],[11,12]])
        self.assertEqual(result['state'],'native_multicolor_path_candidate');self.assertFalse(result['physical_identity_confirmed'])

    def test_gap_is_never_bridged(self):
        rgb,regions=self.fixture();rgb[:,48:52]=20
        self.assertEqual(observe_paths(rgb,regions,[[0,17],[11,12]])[0]['qualifying_hue_families'],0)

    def test_one_color_is_not_two_independent_features(self):
        rgb,regions=self.fixture();rgb[25:28]=20
        self.assertEqual(observe_paths(rgb,regions,[[0,17],[11,12]])[0]['state'],'insufficient_native_color_paths')

    def test_overlapping_families_rejected(self):
        rgb,regions=self.fixture()
        with self.assertRaises(ValueError):observe_paths(rgb,regions,[[0,17],[0,1]])

    def test_circular_palette_grouping(self):
        self.assertEqual(hue_families([17,0,1,9,10]),[[0,1,17],[9,10]])

    def test_input_pixels_unchanged(self):
        rgb,regions=self.fixture();before=rgb.copy();observe_paths(rgb,regions,[[0,17],[11,12]])
        np.testing.assert_array_equal(rgb,before)


if __name__=='__main__':unittest.main()
