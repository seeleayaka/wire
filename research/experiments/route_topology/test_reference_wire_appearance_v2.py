import unittest
import numpy as np
from reference_wire_appearance_v2 import fit_reference,assess


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((60,80,3),[220,30,30],dtype=np.uint8)
        self.mask=np.ones((60,80),bool)
        self.reference=fit_reference(self.rgb,self.mask)

    def test_sparse_colored_reflection_not_reference_wire(self):
        black=np.full_like(self.rgb,30)
        black[20:24,30:40]=[220,30,30]
        metrics,_=assess(black,self.mask,self.reference)
        self.assertEqual(metrics['matched_color_fraction'],1.)
        self.assertEqual(metrics['state'],'insufficient_colored_coverage')

    def test_reference_gain_still_supported(self):
        for gain in [.8,1,1.2]:
            rgb=np.clip(self.rgb.astype(float)*gain,0,255).astype('uint8')
            self.assertEqual(assess(rgb,self.mask,self.reference)[0]['state'],'reference_color_supported')


if __name__=='__main__':unittest.main()
