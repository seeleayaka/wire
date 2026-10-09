import unittest
import numpy as np
from seeded_visible_color import supplement


class SeededVisibleTests(unittest.TestCase):
    def test_recovers_visible_continuation_not_disconnected_color(self):
        rgb=np.zeros((40,60,3),np.uint8);rgb[10:15,5:30]=[255,0,0];rgb[25:30,40:55]=[255,0,0]
        native=np.zeros((40,60),bool);native[10:15,5:12]=True;before=native.copy()
        result,stats=supplement(rgb,native,[0,1,2,3,16,17]);self.assertEqual(int(result.sum()),125)
        self.assertEqual(stats['recovered_RGB_pixels'],90);np.testing.assert_array_equal(native,before)

    def test_does_not_fill_white_label(self):
        rgb=np.zeros((40,60,3),np.uint8);rgb[10:15,5:30]=[255,0,0];rgb[10:15,15:20]=255
        native=np.zeros((40,60),bool);native[10:15,5:12]=True
        result,_=supplement(rgb,native,[0,1,2,3,16,17]);self.assertFalse(result[10:15,15:20].any());self.assertFalse(result[10:15,20:30].any())

    def test_no_seed_and_low_saturation(self):
        rgb=np.full((40,60,3),200,np.uint8);native=np.ones((40,60),bool)
        result,_=supplement(rgb,native,[0,1,2,3,16,17]);self.assertFalse(result.any())


if __name__=='__main__':unittest.main()
