import unittest
import numpy as np
import torch
from paired_boxpool_features import embeddings,footprint_weights
from paired_shape_features import crop_signature,aspect_examples


class BoxPoolTests(unittest.TestCase):
    def test_area_fraction_and_translation(self):
        box=[100,100,180,140];signature=crop_signature(box,1.5)
        weights=footprint_weights(box,signature)
        self.assertAlmostEqual(float(weights.sum()),1.,places=6)
        translated=[v+160 for v in box];other=footprint_weights(translated,tuple(v+160 for v in signature))
        self.assertTrue(torch.equal(weights,other))
        self.assertFalse(torch.equal(weights,footprint_weights(aspect_examples(box)[0],signature)))
        with self.assertRaises(ValueError):footprint_weights([1,1,1,2],signature)

    def test_identical_inputs_one_forward_different_candidate_pooling(self):
        class Encoder:
            calls=0
            def forward_features(self,tensor):
                self.calls+=1;n=len(tensor);patch=torch.zeros((16,16,384))
                patch[:,:,0]=1.;patch[:,:,1]=(torch.arange(16).float()-7.5).square()[:,None]
                return dict(x_norm_clstoken=torch.ones((n,384)),x_norm_patchtokens=patch.reshape(1,256,384).repeat(n,1,1))
        model=Encoder();box=[100,100,180,140];audit={}
        result=embeddings(model,np.ones((300,300,3),np.uint8),[box,aspect_examples(box)[0]],audit=audit)
        self.assertEqual(tuple(result.shape),(2,1536));self.assertEqual(model.calls,1)
        self.assertEqual(audit['unique_crops'],2);self.assertFalse(torch.equal(result[0],result[1]))
        self.assertTrue(torch.equal(result[0,:384],result[1,:384]))
        self.assertEqual(tuple(embeddings(model,np.ones((10,10,3),np.uint8),[]).shape),(0,1536))

if __name__=='__main__':unittest.main()
