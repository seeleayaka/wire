import unittest
import numpy as np
from robust_port_exposure import compensate

class PhotometricContract(unittest.TestCase):
    def setUp(self):
        self.expected=np.random.default_rng(20261004).integers(40,180,size=(1024,1024,3),dtype=np.uint8)
        self.mask=np.ones((1024,1024),dtype=np.float32)
    def test_uniform_exposure_estimation_not_hardcoded_to_one_multiplier(self):
        for gain in (.8,.85,1.15):
            with self.subTest(gain=gain):
                observed=np.rint(self.expected.astype(float)*gain).clip(0,255).astype(np.uint8)
                restored,evidence=compensate(observed,self.expected,self.mask)
                self.assertEqual(evidence['status'],'compensated')
                self.assertLess(np.abs(restored.astype(float)-self.expected).mean(),.5)
    def test_real_changed_region_is_preserved_not_replaced_with_reference(self):
        observed=np.rint(self.expected.astype(float)*.85).astype(np.uint8);observed[:180]=0
        restored,evidence=compensate(observed,self.expected,self.mask)
        self.assertEqual(evidence['status'],'compensated');self.assertTrue((restored[:180]==0).all())
        self.assertLess(np.abs(restored[180:].astype(float)-self.expected[180:]).mean(),.5)
    def test_inputs_remain_unchanged(self):
        observed=np.rint(self.expected.astype(float)*.85).astype(np.uint8)
        a,b,m=observed.copy(),self.expected.copy(),self.mask.copy();compensate(observed,self.expected,self.mask)
        self.assertTrue(np.array_equal(observed,a));self.assertTrue(np.array_equal(self.expected,b));self.assertTrue(np.array_equal(self.mask,m))
    def test_identity_pair_does_not_modify_pixels(self):
        restored,evidence=compensate(self.expected,self.expected,self.mask)
        self.assertEqual(evidence['status'],'identity');self.assertTrue(np.array_equal(restored,self.expected))
    def test_empty_validity_mask_abstains(self):
        restored,evidence=compensate(self.expected,self.expected,np.zeros_like(self.mask))
        self.assertEqual(evidence['status'],'abstained');self.assertTrue(np.array_equal(restored,self.expected))
    def test_severe_exposure_is_not_forcibly_corrected(self):
        observed=np.rint(self.expected.astype(float)*.4).astype(np.uint8);restored,evidence=compensate(observed,self.expected,self.mask)
        self.assertEqual(evidence['status'],'abstained');self.assertTrue(np.array_equal(restored,observed))
    def test_nonuniform_shading_can_fail_closed(self):
        observed=self.expected.copy();observed[:512]=np.rint(self.expected[:512].astype(float)*.75).astype(np.uint8)
        observed[512:]=np.rint(self.expected[512:].astype(float)*1.25).astype(np.uint8)
        restored,evidence=compensate(observed,self.expected,self.mask)
        self.assertEqual(evidence['status'],'abstained');self.assertTrue(np.array_equal(restored,observed))
    def test_bad_shapes_types_or_masks_rejected(self):
        for a,b,m in ((self.expected[:400],self.expected,self.mask),(self.expected.astype(float),self.expected,self.mask),(self.expected,self.expected,self.mask[:400]),(self.expected,self.expected,self.mask*np.nan)):
            with self.assertRaises(ValueError):compensate(a,b,m)
    def test_independent_channel_gain_estimates(self):
        observed=np.rint(self.expected.astype(float)*np.array([.85,1.1,.9])).astype(np.uint8)
        restored,evidence=compensate(observed,self.expected,self.mask);self.assertEqual(evidence['status'],'compensated')
        self.assertLess(np.abs(restored.astype(float)-self.expected).mean(),.5)

if __name__=='__main__':unittest.main()
