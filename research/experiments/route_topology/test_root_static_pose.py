import unittest
import numpy as np
from root_static_pose import normalize


class RootDescriptorTests(unittest.TestCase):
    def test_histogram_amplitude_does_not_change_descriptor(self):
        x=np.arange(1,257,dtype=np.float32).reshape(2,128)
        original=x.copy(); a=normalize(x)
        np.testing.assert_allclose(a,normalize(x*3),atol=1e-7)
        np.testing.assert_allclose((a*a).sum(1),1,atol=1e-6)
        np.testing.assert_array_equal(x,original)

    def test_bad_inputs_fail_closed(self):
        for x in [np.zeros((1,128)),np.ones((1,127)),np.full((1,128),-1),np.full((1,128),np.nan)]:
            with self.assertRaises(ValueError): normalize(x)

    def test_shapes_dtype_and_exact_transform(self):
        x=np.ones((3,128),np.float64)
        a=normalize(x)
        self.assertEqual(a.shape,(3,128)); self.assertEqual(a.dtype,np.float32)
        np.testing.assert_allclose(a,np.sqrt(1/128),atol=1e-7)


if __name__=='__main__': unittest.main()
