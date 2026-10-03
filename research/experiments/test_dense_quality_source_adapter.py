import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import evaluate_dense_quality_source as adapter


class QualitySourceAdapterTests(unittest.TestCase):
    def test_final_real_evaluator_uses_quality_checkpoint_and_restores(self):
        e=adapter.plain.evaluator;before=(e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS,e.main)
        seen=[]
        def verify():
            seen.append(True)
            self.assertEqual(e.TRAIN.name,'dense_quality_20261003')
            self.assertEqual(e.OUT.parent,e.TRAIN)
            self.assertEqual(e.EPOCHS,24)
            self.assertIs(e.decode,adapter.plain.decode)
            self.assertEqual(e.POLICY['candidate_score'],.75)
            self.assertEqual(e.POLICY['minimum_distinct_tiles'],2)
        with patch.object(e,'main',side_effect=verify):adapter.main()
        self.assertEqual(len(seen),1)
        self.assertEqual((e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS,e.main),before)
    def test_nested_failure_restores(self):
        e=adapter.plain.evaluator;before=(e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS,e.main)
        with patch.object(e,'main',side_effect=ValueError('synthetic')):
            with self.assertRaises(ValueError):adapter.main()
        self.assertEqual((e.TRAIN,e.OUT,e.DensePortHead,e.decode,e.frozen_features,e.save,e.EPOCHS,e.main),before)


if __name__=='__main__':unittest.main()
