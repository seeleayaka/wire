import unittest
import numpy as np
from regional_contact import score

class RegionalContactTests(unittest.TestCase):
    def test_nearby_contact_does_not_require_exact_single_pixel(self):
        rgb=np.zeros((21,21,3),np.uint8);rgb[10,10]=255
        support=np.zeros((21,21),bool);support[10,10]=True
        first=score(rgb,support);shifted=np.zeros_like(rgb);shifted[10,12]=255
        self.assertEqual(first,score(shifted,support));self.assertGreater(first,0)
    def test_area_feature_never_modifies_image_or_support(self):
        rgb=np.zeros((21,21,3),np.uint8);rgb[10,10]=255
        support=np.ones((21,21),bool);before=rgb.copy();s=support.copy()
        score(rgb,support);np.testing.assert_array_equal(rgb,before);np.testing.assert_array_equal(support,s)
    def test_color_dark_are_not_positive(self):
        support=np.ones((21,21),bool);rgb=np.zeros((21,21,3),np.uint8);rgb[:,:,0]=255
        self.assertEqual(score(rgb,support),0)
    def test_invalid_support_rejected(self):
        with self.assertRaises(ValueError):score(np.zeros((21,21,3),np.uint8),np.zeros((21,21),bool))
if __name__=='__main__':unittest.main()
