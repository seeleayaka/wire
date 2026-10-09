import unittest
import numpy as np
from tangent_gap_candidates import geometry,match_fragments,describe_fragments,observe_family


def endpoint(i,component,x,y,u,width=4):
    return dict(endpoint_id=i,component=component,point_xy=[x,y],outward_unit=u,width=width)


class TangentGapTests(unittest.TestCase):
    def pair(self):return [endpoint(0,1,30,40,[1,0]),endpoint(1,2,50,40,[-1,0])]
    def test_straight_and_reverse(self):
        a,b=self.pair();g=geometry(a,b);r=geometry(b,a)
        self.assertAlmostEqual(g['score'],r['score'])
        self.assertEqual(match_fragments([a,b])[0]['state'],'fragment_pair_candidate')
        self.assertFalse(g['physical_identity_confirmed'])
    def test_gentle_slope(self):
        a,b=self.pair();b['point_xy'][1]=45
        self.assertIsNotNone(geometry(a,b))
    def test_wrong_direction(self):
        a,b=self.pair();b['outward_unit']=[1,0]
        self.assertIsNone(geometry(a,b))
    def test_width_and_distance_reject(self):
        a,b=self.pair();b['width']=12;self.assertIsNone(geometry(a,b))
        b=self.pair()[1];b['point_xy'][0]=200;self.assertIsNone(geometry(a,b))
    def test_competing_same_color_not_forced(self):
        a,b=self.pair();c=endpoint(2,3,50,42,[-1,0])
        self.assertTrue(all(e['state']=='ambiguous_or_nonreciprocal' for e in match_fragments([a,b,c])))
    def test_same_component_not_an_inferred_gap(self):
        a,b=self.pair();b['component']=a['component'];self.assertIsNone(geometry(a,b))
    def test_raw_pixels_and_unknown_contract(self):
        raw=np.zeros((80,120),bool);raw[38:43,10:42]=True;raw[38:43,56:100]=True
        before=raw.copy();left=np.zeros_like(raw);right=np.zeros_like(raw)
        left[:,10:20]=True;right[:,90:100]=True
        result,skel=observe_family(raw,{'A':left,'B':right})
        self.assertTrue(np.array_equal(raw,before));self.assertTrue((~skel|raw).all())
        self.assertEqual(result['observed_pixels_added'],0)
        self.assertTrue(result['anchor_path_candidates'][0]['contains_inferred_gap'])
        self.assertEqual(result['electrical_continuity'],'not_assessed')
    def test_border_endpoint_excluded(self):
        raw=np.zeros((50,80),bool);raw[20:25,:60]=True
        _,ends,_,_=describe_fragments(raw)
        self.assertTrue(all(e['point_xy'][0]>=2 for e in ends))
    def test_sharp_corner_not_forced(self):
        a,b=self.pair();b['point_xy']=[30,60]
        self.assertIsNone(geometry(a,b))
    def test_branch_not_an_endpoint(self):
        raw=np.zeros((80,80),bool);raw[40,10:70]=True;raw[10:41,40]=True
        _,ends,_,_=describe_fragments(raw)
        self.assertFalse(any(e['point_xy']==[40.,40.] for e in ends))
    def test_empty_mask_and_no_vote_inflation(self):
        raw=np.zeros((40,40),bool)
        result,_=observe_family(raw,{'A':raw.copy(),'B':raw.copy()})
        self.assertEqual(result['anchor_path_candidates'],[])
        self.assertFalse(result['physical_identity_confirmed'])
    def test_translation_invariance(self):
        a,b=self.pair();first=geometry(a,b)
        for e in [a,b]:e['point_xy']=[v+100 for v in e['point_xy']]
        self.assertAlmostEqual(first['score'],geometry(a,b)['score'])


if __name__=='__main__':unittest.main()
