import unittest
import numpy as np
import torch
from paired_core_ring_missing import descriptor,force_missing_abstention,fold_standardization,DIMENSIONS


class MissingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((512,512,3),120,np.uint8)
        self.mask=np.ones((512,512),bool);self.box=[200,200,300,300]

    def test_missing_geometry_explicit_not_removed(self):
        x=descriptor(self.image,self.image,np.zeros_like(self.mask),self.box)
        np.testing.assert_array_equal(x,np.zeros(DIMENSIONS,np.float32))
        self.assertEqual(x.shape,(DIMENSIONS,))

    def test_valid_descriptor_present(self):
        x=descriptor(self.image,self.image,self.mask,self.box)
        self.assertEqual(x[-1],1)

    def test_invalid_input_not_silently_missing(self):
        with self.assertRaises(ValueError):descriptor(self.image,self.image,self.mask,[0,0,float('nan'),20])
        with self.assertRaises(ValueError):descriptor(self.image.astype(float),self.image,self.mask,self.box)

    def test_missing_never_positive_preserves_existing_scores(self):
        scores=torch.tensor([[.001,.998,.001],[.001,.001,.998]])
        original=scores.clone();d=torch.zeros((2,DIMENSIONS));d[1,-1]=1
        actual=force_missing_abstention(scores,d)
        torch.testing.assert_close(actual[0],torch.tensor([1.,0.,0.]))
        torch.testing.assert_close(actual[1],scores[1]);torch.testing.assert_close(scores,original)

    def test_invalid_probabilities_and_flags_rejected(self):
        d=torch.zeros((1,DIMENSIONS))
        with self.assertRaises(ValueError):force_missing_abstention(torch.tensor([[2.,0.,0.]]),d)
        d[0,-1]=.5
        with self.assertRaises(ValueError):force_missing_abstention(torch.tensor([[1.,0.,0.]]),d)
        d[0,-1]=0;d[0,1]=.1
        with self.assertRaises(ValueError):force_missing_abstention(torch.tensor([[1.,0.,0.]]),d)

    def test_scaling_uses_training_only(self):
        a=torch.stack((torch.zeros(DIMENSIONS),torch.ones(DIMENSIONS)))
        x,y,mean,std=fold_standardization(a,torch.ones((1,DIMENSIONS))*999)
        torch.testing.assert_close(mean,torch.ones(DIMENSIONS)*.5)
        torch.testing.assert_close(std,torch.ones(DIMENSIONS)*.5)
        torch.testing.assert_close(y,torch.ones((1,DIMENSIONS))*5)


if __name__=='__main__':unittest.main()
