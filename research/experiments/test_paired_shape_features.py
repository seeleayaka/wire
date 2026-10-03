import unittest
import torch
from paired_shape_features import crop_signature,shape_vector,aspect_examples,shaped_features


class ShapeFeatures(unittest.TestCase):
    def test_translation_scale_invariance_without_position(self):
        self.assertEqual(shape_vector([10,20,90,60]),shape_vector([200,300,360,380]))
        with self.assertRaises(ValueError):shape_vector([1,1,1,2])

    def test_short_axis_changes_preserve_original_crop_but_change_descriptor(self):
        box=[100,100,180,140]
        for altered in aspect_examples(box):
            for scale in (1.5,3.):self.assertEqual(crop_signature(box,scale),crop_signature(altered,scale))
            self.assertNotEqual(shape_vector(box),shape_vector(altered))
        self.assertEqual(tuple(shaped_features(torch.zeros((1,6144)),[box]).shape),(1,6147))

if __name__=='__main__':unittest.main()
