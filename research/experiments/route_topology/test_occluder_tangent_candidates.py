import unittest
import numpy as np
from soft_tangent_candidates import match_fragments
from occluder_tangent_candidates import augment_edges


def end(i,p,u):
    return dict(endpoint_id=i,component=i+1,point_xy=p,outward_unit=u,width=4.,direction_reliability=1.)


class OccluderTests(unittest.TestCase):
    def scene(self):
        rgb=np.full((80,100,3),[20,100,20],dtype=np.uint8);raw=np.zeros((80,100),bool)
        raw[38:43,5:21]=True;raw[38:43,40:56]=True;raw[43:48,40:56]=True
        ends=[end(0,[20,40],[1,0]),end(1,[40,40],[-1,0]),end(2,[40,45],[-1,0])]
        return rgb,raw,ends

    def test_shared_label_resolves_only_supported_competition(self):
        rgb,raw,ends=self.scene();rgb[39:42,23:38]=200
        old=match_fragments(ends);self.assertTrue(all(e['state']!='fragment_pair_candidate' for e in old))
        result=augment_edges(ends,old,rgb,raw)
        added=[e for e in result if e['state']=='occluder_supported_fragment_candidate']
        self.assertEqual([e['endpoint_ids'] for e in added],[[0,1]])

    def test_no_obstacle_does_not_resolve_competition(self):
        rgb,raw,ends=self.scene();result=augment_edges(ends,match_fragments(ends),rgb,raw)
        self.assertFalse(any(e['state']=='occluder_supported_fragment_candidate' for e in result))

    def test_uniform_background_is_not_obstacle(self):
        rgb,raw,ends=self.scene();rgb[:]=0
        result=augment_edges(ends,match_fragments(ends),rgb,raw)
        self.assertFalse(any(e['state']=='occluder_supported_fragment_candidate' for e in result))

    def test_prior_edges_preserved_and_no_pixels_added(self):
        rgb,raw,ends=self.scene();old=match_fragments(ends[:2]);before=raw.copy()
        result=augment_edges(ends[:2],old,rgb,raw)
        self.assertEqual(result[0]['state'],old[0]['state']);self.assertEqual(result[0]['score'],old[0]['score'])
        np.testing.assert_array_equal(raw,before)


if __name__=='__main__':unittest.main()
