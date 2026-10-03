import unittest
import numpy as np
import torch
from paired_core_ring import descriptor,normalize_pair,fold_standardization,DIMENSIONS


class CoreRingTests(unittest.TestCase):
    def setUp(self):
        self.expected=np.full((1024,1024,3),120,np.uint8)
        self.valid=np.ones((1024,1024),bool)
        self.box=[400,400,600,600]

    def test_equal_pair_no_residual(self):
        x=descriptor(self.expected,self.expected,self.valid,self.box)
        self.assertEqual(x.shape,(DIMENSIONS,))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(x[:16]==0))
        self.assertTrue(np.all(x[36:43]==0))

    def test_body_vs_ring_separated(self):
        a=self.expected.copy();a[400:600,400:600]=200
        b=self.expected.copy();b[200:400,200:800]=200
        core=descriptor(a,self.expected,self.valid,self.box)
        ring=descriptor(b,self.expected,self.valid,self.box)
        self.assertGreater(core[:3].mean(),.3)
        self.assertLess(core[18:21].mean(),.01)
        self.assertLess(ring[:3].mean(),.01)
        self.assertGreater(ring[18:21].mean(),.03)

    def test_translation_equivariance(self):
        rng=np.random.default_rng(1);a=rng.integers(0,256,(512,512,3),dtype=np.uint8)
        b=rng.integers(0,256,(512,512,3),dtype=np.uint8);m=np.ones((512,512),bool)
        x=descriptor(a,b,m,[190,190,290,290])
        y=descriptor(np.pad(a,((31,31),(47,47),(0,0))),np.pad(b,((31,31),(47,47),(0,0))),np.pad(m,((31,31),(47,47))),[237,221,337,321])
        np.testing.assert_allclose(x,y,atol=1e-6)

    def test_color_channels_permuted(self):
        a=self.expected.copy();a[400:600,400:600]=[180,150,130]
        x=descriptor(a,self.expected,self.valid,self.box)
        y=descriptor(a[...,::-1].copy(),self.expected[...,::-1].copy(),self.valid,self.box)
        np.testing.assert_allclose(x[:3][::-1],y[:3],atol=1e-6)
        np.testing.assert_allclose(x[36:40],y[36:40],atol=1e-6)

    def test_inputs_immutable(self):
        a=self.expected.copy();m=self.valid.copy();box=self.box.copy()
        descriptor(a,self.expected,m,box)
        np.testing.assert_array_equal(a,self.expected)
        np.testing.assert_array_equal(m,self.valid)
        self.assertEqual(box,self.box)

    def test_bad_coverage_and_geometry_fail_closed(self):
        with self.assertRaises(ValueError):descriptor(self.expected,self.expected,np.zeros_like(self.valid),self.box)
        for box in ([0,0,1,1],[1,1,float('nan'),20],[-100,400,50,600]):
            with self.assertRaises(ValueError):descriptor(self.expected,self.expected,self.valid,box)

    def test_thin_component_still_sampled(self):
        x=descriptor(self.expected,self.expected,self.valid,[490,400,494,600])
        self.assertEqual(x.shape,(DIMENSIONS,))

    def test_normalization_does_not_replace_local_defect(self):
        a=np.full_like(self.expected,96);a[400:600,400:600]=64
        copy,meta=normalize_pair(a,self.expected,self.valid)
        self.assertEqual(meta['status'],'compensated')
        self.assertGreater(descriptor(copy,self.expected,self.valid,self.box)[:3].mean(),.1)
        self.assertTrue(np.all(a[400:600,400:600]==64))

    def test_standardization_train_only(self):
        a=torch.stack([torch.zeros(DIMENSIONS),torch.ones(DIMENSIONS)])
        _,y,mean,std=fold_standardization(a,torch.ones((1,DIMENSIONS))*100)
        self.assertTrue(torch.equal(mean,torch.ones(DIMENSIONS)*.5))
        self.assertTrue(torch.equal(std,torch.ones(DIMENSIONS)*.5))
        self.assertTrue(torch.equal(y,torch.ones((1,DIMENSIONS))*5))

    def test_constant_and_nonfinite_features(self):
        a=torch.zeros((2,DIMENSIONS));x,y,_,s=fold_standardization(a,a)
        self.assertTrue(torch.isfinite(x).all());self.assertTrue((s>=.01).all())
        a[0,0]=float('nan')
        with self.assertRaises(ValueError):fold_standardization(a,a)


if __name__=='__main__':unittest.main()
