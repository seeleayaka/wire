import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
import numpy as np
import evaluate_dense_pixel_augmented_source as adapter


class AugmentedSourceAdapterTests(unittest.TestCase):
    def test_same_decoder_features_and_source_gates_restored(self):
        e=adapter.evaluator
        before=(e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS)
        policy=dict(e.POLICY)
        def inspect():
            self.assertEqual(e.EPOCHS,24)
            self.assertEqual(e.TRAIN.name,'dense_pixel_augmented_20261003')
            self.assertIs(e.decode,adapter.decode)
            self.assertEqual(e.POLICY,policy)
            with patch.object(adapter,'semantic_features',return_value=torch.zeros((1,384,32,32))):
                features=e.frozen_features(object(),[np.zeros((640,640,3),dtype=np.uint8)])
            self.assertEqual(tuple(features['rgb'].shape),(1,3,448,448))
            self.assertEqual(tuple(features['semantic'].shape),(1,384,32,32))
            head=e.DensePortHead();torch.set_num_threads(2)
            self.assertEqual(tuple(head(features)[0].shape),(1,2,56,56))
        with patch.object(e,'main',side_effect=inspect):adapter.main()
        self.assertEqual((e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS),before)
    def test_failure_restores_evaluator_globals(self):
        e=adapter.evaluator;before=(e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS)
        with patch.object(e,'main',side_effect=ValueError('synthetic fail')):
            with self.assertRaisesRegex(ValueError,'synthetic fail'):adapter.main()
        self.assertEqual((e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS),before)


if __name__=='__main__':unittest.main()
