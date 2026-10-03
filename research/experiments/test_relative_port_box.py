import math
import unittest
from relative_port_box import encode,decode,bounded


class RelativeGeometry(unittest.TestCase):
    def test_identity(self):
        box=[33.,42.,93.,62.]
        self.assertEqual(decode(box,[0,0,0,0]),box)

    def test_safe_delta_roundtrip(self):
        box=[120,230,160,310];target=[118,236,162,308]
        result=decode(box,encode(box,target))
        for a,b in zip(result,target):self.assertAlmostEqual(a,b,places=10)

    def test_translation_and_anisotropic_scaling_equivariance(self):
        box=[120,230,160,310];delta=[.1,-.13,math.log(1.1),math.log(.9)]
        result=decode(box,delta)
        for sx,sy,dx,dy in ((3,2,150,-30),(.5,.75,-90,37)):
            transform=lambda a:[a[0]*sx+dx,a[1]*sy+dy,a[2]*sx+dx,a[3]*sy+dy]
            prediction=decode(transform(box),delta)
            for a,b in zip(prediction,transform(result)):self.assertAlmostEqual(a,b,places=9)
            for a,b in zip(encode(transform(box),prediction),delta):self.assertAlmostEqual(a,b,places=9)

    def test_predictions_are_bounded_not_arbitrary_absolute_positions(self):
        delta=bounded([100,-100,100,-100]);self.assertEqual(delta[:2],[.15,-.15])
        self.assertAlmostEqual(math.exp(delta[2]),1.2);self.assertAlmostEqual(math.exp(delta[3]),.8)

    def test_inputs_unmodified(self):
        box=[12,22,42,62];delta=[.1,-.1,.1,-.1];a,b=box.copy(),delta.copy();decode(box,delta)
        self.assertEqual(a,box);self.assertEqual(b,delta)

    def test_invalid_geometry_and_nonfinite_offsets_fail_closed(self):
        for box in ([0,0,0,1],[0,0,1,float('nan')],[0,0,1]):
            with self.assertRaises(ValueError):decode(box,[0,0,0,0])
        for delta in ([0,0,float('inf'),0],[0,0,0]):
            with self.assertRaises(ValueError):decode([1,2,3,4],delta)


if __name__=='__main__':unittest.main()
