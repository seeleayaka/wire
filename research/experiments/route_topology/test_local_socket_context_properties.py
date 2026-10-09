"""Generic coordinate properties, no image fixtures or GT parameters."""
import unittest
import numpy as np
from local_socket_crop import context_box
from local_socket_box import positive_box


class ContextProperties(unittest.TestCase):
    def test_two_hundred_affine_contexts(self):
        rng=np.random.default_rng(0);anchor=[800,700,900,750]
        for _ in range(200):
            angle=rng.uniform(-.2,.2);scale=rng.uniform(.8,1.2);shear=rng.uniform(-.05,.05)
            linear=scale*np.array([[np.cos(angle),-np.sin(angle)+shear],[np.sin(angle),np.cos(angle)]])
            matrix=np.eye(3);matrix[:2,:2]=linear;matrix[:2,2]=rng.uniform(-50,50,2)
            context=context_box(anchor,matrix,[2000,2000]);box,norm=positive_box(anchor,matrix,context)
            self.assertTrue(context[0]<=box[0]<box[2]<=context[2])
            self.assertTrue(context[1]<=box[1]<box[3]<=context[3])
            self.assertTrue(all(0<v<1 for v in norm))
            cx,cy,w,h=norm
            self.assertGreaterEqual(cx-w/2,0);self.assertLessEqual(cx+w/2,1)
            self.assertGreaterEqual(cy-h/2,0);self.assertLessEqual(cy+h/2,1)
            inverse=np.linalg.inv(matrix)
            for x,y in [(800,700),(900,700),(900,750),(800,750)]:
                q=inverse@np.array([x,y,1.]);xx,yy=q[:2]/q[2]
                self.assertTrue(box[0]<=xx<=box[2] and box[1]<=yy<=box[3])


if __name__=='__main__':unittest.main()
