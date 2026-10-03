import unittest
import numpy as np
from tools.run_fine_cnn_validation import nearby_distance
from tools.probe_local_normal_bank import local_distance


class FineDistanceTests(unittest.TestCase):
    def test_pre_normalized_bank_matches_original(self):
        rng=np.random.default_rng(20260929)
        bank=rng.normal(size=(5,4,6,8)).astype(np.float32)
        query=rng.normal(size=(4,6,8)).astype(np.float32)
        normalized=bank/np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-8)
        for radius in (0,1,2,5):
            for k in (1,3,5):
                with self.subTest(radius=radius,k=k):
                    np.testing.assert_allclose(nearby_distance(query,normalized,k=k,radius=radius),
                                               local_distance(query,bank,k=k,radius=radius),rtol=1e-6,atol=1e-6)
    def test_identical_and_zero_vectors_match(self):
        for constant in (0.,1.):
            bank=np.full((3,3,3,4),constant,dtype=np.float32);query=bank[0].copy()
            normalized=bank/np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-8)
            np.testing.assert_allclose(nearby_distance(query,normalized),local_distance(query,bank,radius=2),atol=1e-6)


if __name__=='__main__':unittest.main()
