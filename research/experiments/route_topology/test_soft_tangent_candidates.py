import math
import unittest
import numpy as np
from soft_tangent_candidates import geometry,match_fragments,direction_evidence,observe_family


def endpoint(i,c,p,u,w=4,reliability=1):
    return dict(endpoint_id=i,component=c,point_xy=p,outward_unit=u,width=w,direction_reliability=reliability)


class SoftTangentTests(unittest.TestCase):
    def pair(self):return endpoint(0,1,[20,40],[1,0]),endpoint(1,2,[40,40],[-1,0])
    def test_width_difference_not_hard_rejected(self):
        a,b=self.pair();b['width']=9
        self.assertEqual(match_fragments([a,b])[0]['state'],'fragment_pair_candidate')
    def test_moderate_angle_not_hard_rejected(self):
        a,b=self.pair();a['outward_unit']=[math.cos(.8),math.sin(.8)]
        self.assertEqual(match_fragments([a,b])[0]['state'],'fragment_pair_candidate')
    def test_competition_remains_unknown(self):
        a,b=self.pair();c=endpoint(2,3,[40,42],[-1,0])
        self.assertTrue(all(e['state']!='fragment_pair_candidate' for e in match_fragments([a,b,c])))
    def test_reverse_and_far_reject(self):
        a,b=self.pair();b['outward_unit']=[1,0];self.assertIsNone(geometry(a,b))
        b=self.pair()[1];b['point_xy']=[400,40];self.assertIsNone(geometry(a,b))
    def test_reversal_and_translation_invariance(self):
        a,b=self.pair();first=geometry(a,b)['score'];self.assertAlmostEqual(first,geometry(b,a)['score'])
        for e in [a,b]:e['point_xy']=[v+100 for v in e['point_xy']]
        self.assertAlmostEqual(first,geometry(a,b)['score'])
    def test_multiscale_support(self):
        a,_=self.pair();a['support_points_xy']=[[20-i,40] for i in range(13)]
        before=dict(a);d=direction_evidence(a)
        self.assertEqual(d['direction_scales'],3);self.assertAlmostEqual(d['direction_reliability'],1)
        self.assertEqual(a,before)
    def test_short_support_less_reliable(self):
        a,_=self.pair();a['support_points_xy']=[[20-i,40] for i in range(6)]
        self.assertAlmostEqual(direction_evidence(a)['direction_reliability'],1/3)
    def test_pixels_and_inferred_contract(self):
        raw=np.zeros((80,120),bool);raw[38:43,10:42]=True;raw[38:43,56:100]=True
        before=raw.copy();left=np.zeros_like(raw);right=np.zeros_like(raw)
        left[:,10:20]=True;right[:,90:100]=True
        result,skel=observe_family(raw,{'A':left,'B':right})
        self.assertTrue(np.array_equal(raw,before));self.assertTrue((~skel|raw).all())
        self.assertTrue(result['anchor_path_candidates'][0]['contains_inferred_gap'])
        self.assertEqual(result['observed_pixels_added'],0);self.assertFalse(result['physical_identity_confirmed'])


if __name__=='__main__':unittest.main()
