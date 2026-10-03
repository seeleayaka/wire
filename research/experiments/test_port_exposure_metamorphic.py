"""Photo-independent photometric metamorphic checks, never threshold fitting."""
import unittest
import numpy as np
from robust_port_exposure import compensate


class ExposureMetamorphicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=np.random.default_rng(7).integers(45,175,size=(1024,1024,3),dtype=np.uint8)
        cls.mask=np.ones((1024,1024),dtype=bool)

    def test_seeded_channel_exposure_range_not_one_photo_or_fixed_gain(self):
        for gains in np.random.default_rng(19).uniform(.76,1.28,(20,3)):
            observed=np.rint(self.reference.astype(float)*gains).astype(np.uint8)
            result,evidence=compensate(observed,self.reference,self.mask)
            self.assertIn(evidence['status'],('compensated','identity'))
            # A deliberate near-identity skip is not expected to equal reference.
            if evidence['status']=='compensated':
                self.assertLess(np.abs(result.astype(float)-self.reference).mean(),.6)

    def test_channel_permutation_commutes_with_compensation(self):
        image=np.rint(self.reference.astype(float)*[.83,1.2,.91]).astype(np.uint8)
        result,first=compensate(image,self.reference,self.mask)
        for order in ([2,0,1],[1,2,0]):
            swapped,second=compensate(image[:,:,order],self.reference[:,:,order],self.mask)
            self.assertTrue(np.array_equal(swapped,result[:,:,order]))
            self.assertEqual(second['channel_gain'],[first['channel_gain'][i] for i in order])

    def test_nonzero_structural_change_is_not_replaced_with_reference(self):
        original=np.full((1024,1024,3),120,dtype=np.uint8)
        observed=np.full_like(original,102)
        observed[300:600,320:620]=51
        restored,evidence=compensate(observed,original,self.mask)
        self.assertEqual(evidence['status'],'compensated')
        self.assertTrue((restored[300:600,320:620]==60).all())
        self.assertTrue((restored[:250]==120).all())
        self.assertTrue((restored[300:600,320:620]!=original[300:600,320:620]).all())

    def test_compensated_result_is_only_global_channel_scaling(self):
        observed=np.rint(self.reference.astype(float)*.82).astype(np.uint8)
        observed[333:503,213:493]=np.random.default_rng(31).integers(0,255,(170,280,3),dtype=np.uint8)
        result,evidence=compensate(observed,self.reference,self.mask)
        self.assertEqual(evidence['status'],'compensated')
        independently_scaled=np.clip(np.rint(observed.astype(np.float32)*np.array(evidence['channel_gain'],dtype=np.float32)),0,255).astype(np.uint8)
        self.assertTrue(np.array_equal(result,independently_scaled))
        for channel in range(3):
            # No pixel rearrangement, local interpolation, inpainting or mask warp.
            values=np.arange(256,dtype=np.float32)
            mapping=np.clip(np.rint(values*np.float32(evidence['channel_gain'][channel])),0,255).astype(np.uint8)
            self.assertTrue(np.array_equal(result[:,:,channel],mapping[observed[:,:,channel]]))
            self.assertTrue((np.diff(mapping.astype(int))>=0).all())

    def test_invalid_outside_mask_pixels_cannot_influence_estimated_gain(self):
        mask=self.mask.copy();mask[:128]=False
        observed=np.rint(self.reference.astype(float)*.82).astype(np.uint8)
        first,a=compensate(observed,self.reference,mask)
        changed=observed.copy();changed[:128]=235
        second,b=compensate(changed,self.reference,mask)
        self.assertEqual(a,b)
        self.assertTrue(np.array_equal(first[128:],second[128:]))

    def test_small_images_abstain_instead_of_extrapolating_crop_gain(self):
        observed=self.reference[:224,:224]
        result,evidence=compensate(observed,observed,self.mask[:224,:224])
        self.assertEqual(evidence['status'],'abstained')
        self.assertTrue(np.array_equal(result,observed))


if __name__=='__main__':unittest.main()
