import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,'E:/PythonProject10')
import cv2
import numpy as np
import torch
from paired_port_semantics import paired_features,expected_in_source,valid_boxes,CachedReferenceSIFT


class PairedSemanticsTests(unittest.TestCase):
    def test_no_metadata_channels_and_equal_pair_zero_difference(self):
        a=torch.ones((2,1536));f=paired_features(a,a)
        self.assertEqual(tuple(f.shape),(2,6144));self.assertEqual(float(f[:,3072:4608].abs().sum()),0)
        with self.assertRaises(ValueError):paired_features(a,torch.ones((1,1536)))
        a[0,0]=float('nan')
        with self.assertRaises(ValueError):paired_features(a,a)
    def test_inverse_reference_warp_direction_and_coverage(self):
        reference=np.zeros((200,300,3),np.uint8);reference[80:100,100:120]=255
        matrix=np.array([[1,0,10],[0,1,0],[0,0,1]],float)
        expected,mask=expected_in_source(reference,matrix,[200,300])
        self.assertEqual(int(expected[85,95,0]),255);self.assertFalse(mask[:,-1].any())
        self.assertEqual(valid_boxes([[80,70,120,110]],mask),[0]);self.assertEqual(valid_boxes([[270,70,295,110]],mask),[])
        with self.assertRaises(ValueError):expected_in_source(reference,np.zeros((3,3)),[200,300])
    def test_reference_cache_and_restore_on_failure(self):
        reference=np.zeros((32,32,3),np.uint8);count=[]
        class Fake:
            def detectAndCompute(self,image,mask):count.append(True);return [],np.ones((1,128),np.float32)
        with patch.object(cv2,'SIFT_create',side_effect=lambda *a,**kw:Fake()) as original:
            with self.assertRaises(ValueError):
                with CachedReferenceSIFT(reference) as cache:
                    for _ in range(2):cv2.SIFT_create(nfeatures=9000,contrastThreshold=.014,edgeThreshold=12).detectAndCompute(np.zeros((32,32),np.uint8),None)
                    self.assertEqual(cache.cache_hits,1);raise ValueError('synthetic')
            self.assertIs(cv2.SIFT_create,original)
        self.assertEqual(len(count),1)


if __name__=='__main__':unittest.main()
