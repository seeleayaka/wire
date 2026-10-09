import unittest
import numpy as np
from reference_wire_appearance_v3 import fit_reference,assess


class FiniteWidthTests(unittest.TestCase):
    def test_filling_mask_has_finite_width_and_original_shape(self):
        rgb=np.full((60,80,3),[220,30,30],dtype=np.uint8);mask=np.ones((60,80),bool)
        ref=fit_reference(rgb,mask)
        result,pixels=assess(rgb,mask,ref)
        self.assertTrue(np.isfinite(ref['width_p95']))
        self.assertTrue(np.isfinite(result['median_width_reference_pixels']))
        self.assertEqual(pixels.shape,mask.shape)

    def test_black_region_with_sparse_reference_color_abstains(self):
        rgb=np.full((60,80,3),[220,30,30],dtype=np.uint8);mask=np.ones((60,80),bool)
        ref=fit_reference(rgb,mask)
        black=np.full_like(rgb,30);black[20:24,30:40]=[220,30,30]
        self.assertEqual(assess(black,mask,ref)[0]['state'],'insufficient_colored_coverage')


if __name__=='__main__':unittest.main()
