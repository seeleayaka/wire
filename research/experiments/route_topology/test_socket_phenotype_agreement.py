import unittest
import numpy as np
from socket_phenotype_agreement import feature_view,agreement


class AgreementTests(unittest.TestCase):
    def test_partition_exact(self):
        x=np.arange(320,dtype=np.float64)
        np.testing.assert_array_equal(feature_view(x,'color')+feature_view(x,'edge'),x)

    def test_mismatch_and_abstention(self):
        for labels in [[1,0,1],[1,None,1],[None,None,None]]:
            result=agreement({k:{'visual_label_candidate':v} for k,v in zip(['combined','color','edge'],labels)})
            self.assertIsNone(result['visual_label_candidate'])
            self.assertEqual(result['new_confirmed_connections'],0)

    def test_agreement_single_source_not_three_votes(self):
        r=agreement({k:{'visual_label_candidate':1} for k in ['combined','color','edge']})
        self.assertEqual(r['visual_label_candidate'],1);self.assertEqual(r['independent_observer_count'],1)

    def test_missing_view_rejected(self):
        with self.assertRaises(ValueError):agreement({'combined':{'visual_label_candidate':1}})


if __name__=='__main__':unittest.main()
