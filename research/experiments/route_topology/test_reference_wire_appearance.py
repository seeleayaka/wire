import unittest
import numpy as np
from reference_wire_appearance import fit_reference,assess


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.zeros((80,120,3),dtype=np.uint8)
        self.mask=np.zeros((80,120),dtype=bool)
        for y,color in [(20,(220,30,30)),(30,(230,200,30)),(40,(30,40,220))]:
            self.rgb[y:y+4,20:100]=color
            self.mask[y:y+4,20:100]=True
        self.ref=fit_reference(self.rgb,self.mask)

    def test_reference_and_gain_preserved(self):
        for gain in [.8,1.,1.2]:
            image=np.clip(self.rgb.astype(float)*gain,0,255).astype('uint8')
            self.assertEqual(assess(image,self.mask,self.ref)[0]['state'],'reference_color_supported')

    def test_black_or_gray_not_declared_reference_wire(self):
        rgb=np.full_like(self.rgb,30)
        self.assertEqual(assess(rgb,self.mask,self.ref)[0]['state'],'insufficient_colored_evidence')

    def test_position_translation_and_no_input_mutation(self):
        before=self.rgb.copy();mask_before=self.mask.copy()
        shifted=np.roll(self.rgb,10,axis=1);shift_mask=np.roll(self.mask,10,axis=1)
        self.assertEqual(assess(shifted,shift_mask,self.ref)[0]['state'],'reference_color_supported')
        np.testing.assert_array_equal(self.rgb,before);np.testing.assert_array_equal(self.mask,mask_before)

    def test_green_different_color(self):
        image=self.rgb.copy();image[self.mask]=[30,220,30]
        self.assertEqual(assess(image,self.mask,self.ref)[0]['state'],'appearance_mismatch_or_overwide')

    def test_black_reference_abstains(self):
        with self.assertRaises(ValueError):fit_reference(np.zeros_like(self.rgb),self.mask)

    def test_identical_looking_wire_remains_identity_unknown(self):
        self.assertTrue(assess(self.rgb,self.mask,self.ref)[0]['appearance_only_not_physical_identity'])


if __name__=='__main__':unittest.main()
