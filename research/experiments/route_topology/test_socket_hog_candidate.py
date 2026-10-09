import unittest
import numpy as np
from socket_hog_candidate import descriptor, resolve


class ShapeCandidateTests(unittest.TestCase):
    def test_shape_feature_deterministic_and_does_not_mutate_pixels(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0,256,(50,100,3),dtype=np.uint8); original = rgb.copy()
        a = descriptor(rgb); b = descriptor(rgb)
        self.assertEqual(a.shape,(1980,)); np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(rgb,original)

    def test_preserves_every_supported_old_label_and_never_rescues_exposed(self):
        for old in [0,1]: self.assertEqual(resolve(old,[None]*9,[0]*9),old)
        self.assertEqual(resolve(None,[0]*9,[0]*9),None)
        self.assertEqual(resolve(None,[1]*9,[1]*9),1)

    def test_one_offset_disagreement_or_unknown_retains_unknown(self):
        for index in range(9):
            for value in [0,None]:
                labels=[1]*9; labels[index]=value
                self.assertIsNone(resolve(None,labels,[1]*9))
                self.assertIsNone(resolve(None,[1]*9,labels))

    def test_bad_labels_and_missing_pixels_rejected(self):
        for shape in [(49,100,3),(50,99,3),(50,100)]:
            with self.assertRaises(ValueError): descriptor(np.zeros(shape,np.uint8))
        with self.assertRaises(ValueError): resolve(True,[1]*9,[1]*9)
        with self.assertRaises(ValueError): resolve(None,[True]*9,[1]*9)


if __name__=='__main__': unittest.main()
