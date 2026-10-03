import unittest
import numpy as np
from tools.probe_local_normal_bank import local_distance

class LocalBankTests(unittest.TestCase):
    def test_identical_normal_is_zero_and_orthogonal_anomaly_remains(self):
        q=np.zeros((2,3,2),dtype=np.float32); q[...,0]=1
        bank=np.stack([q]*3)
        np.testing.assert_allclose(local_distance(q,bank),0,atol=1e-6)
        anomalous=q.copy(); anomalous[0,0]=[0,1]
        self.assertAlmostEqual(float(local_distance(anomalous,bank)[0,0]),1)

    def test_k_requires_distinct_normal_images(self):
        q=np.zeros((2,3,2),dtype=np.float32); q[...,0]=1
        wrong=q.copy(); wrong[...,0]=0; wrong[...,1]=1
        score=local_distance(q,np.stack([q,wrong,wrong]),k=3,radius=0)
        np.testing.assert_allclose(score,2/3,atol=1e-6)

    def test_spatial_neighborhood_does_not_wrap_image_edges(self):
        q=np.zeros((2,4,2),dtype=np.float32); q[...,0]=1
        bank=q.copy(); bank[0,3]=[0,1]
        q[0,0]=[0,1]
        self.assertAlmostEqual(float(local_distance(q,bank[None],k=1,radius=1)[0,0]),1)

if __name__=='__main__':
    unittest.main()
