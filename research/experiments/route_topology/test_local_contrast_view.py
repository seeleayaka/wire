import unittest
import numpy as np
from local_contrast_view import contrast_view,POLICY

class ContrastTests(unittest.TestCase):
    def test_no_input_mutation_shape_change_or_nondeterminism(self):
        rgb=np.random.default_rng(0).integers(0,256,(40,63,3),dtype=np.uint8);before=rgb.copy()
        a=contrast_view(rgb);b=contrast_view(rgb)
        np.testing.assert_array_equal(rgb,before);np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,rgb.shape);self.assertEqual(a.dtype,rgb.dtype)

    def test_no_geometric_transport_or_extra_model_vote(self):
        self.assertFalse(POLICY['geometric_resampling']);self.assertFalse(POLICY['mask_morphology'])
        self.assertFalse(POLICY['independent_observer'])

    def test_invalid_frames_rejected(self):
        for rgb in [np.zeros((20,20),np.uint8),np.zeros((20,20,3),float),np.zeros((3,3,3),np.uint8)]:
            with self.assertRaises(ValueError):contrast_view(rgb)

if __name__=='__main__':unittest.main()
