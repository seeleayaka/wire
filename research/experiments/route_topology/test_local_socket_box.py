import unittest
import numpy as np
from local_socket_box import positive_box


class SpatialCueTests(unittest.TestCase):
    def test_reference_fixed_relative(self):
        box,n=positive_box([100,100,200,150],np.eye(3),[0,0,300,250])
        self.assertEqual(box,[50,50,250,200]);np.testing.assert_allclose(n,[.5,.5,2/3,.6])

    def test_translation_invariant(self):
        m=np.array([[1,0,-20],[0,1,-30],[0,0,1.]])
        box,n=positive_box([100,100,200,150],m,[20,30,320,280])
        self.assertEqual(box,[70,80,270,230]);np.testing.assert_allclose(n,[.5,.5,2/3,.6])

    def test_no_clipping(self):
        with self.assertRaises(ValueError):positive_box([100,100,200,150],np.eye(3),[100,100,200,150])


if __name__=='__main__':unittest.main()
