import unittest
from tangent_color_penalty import geometry,match
from test_soft_tangent_candidates import endpoint


def colored(e,bin):
    e['color_histogram']=[float(i==bin) for i in range(18)];return e


class ColorPenaltyTests(unittest.TestCase):
    def test_never_lowers_geometry_score(self):
        a=colored(endpoint(0,1,[20,40],[1,0]),0);b=colored(endpoint(1,2,[40,40],[-1,0]),3)
        e=geometry(a,b);self.assertAlmostEqual(e['color_distance'],1)
        self.assertGreaterEqual(e['score'],e['base_geometry_score'])
    def test_color_resolves_equal_geometry(self):
        a=colored(endpoint(0,1,[20,40],[1,0]),0)
        b=colored(endpoint(1,2,[40,40],[-1,0]),0);c=colored(endpoint(2,3,[40,42],[-1,0]),3)
        selected=[e for e in match([a,b,c]) if e['state']=='fragment_pair_candidate']
        self.assertEqual([e['endpoint_ids'] for e in selected],[[0,1]])
    def test_same_color_competition_kept(self):
        ends=[colored(endpoint(0,1,[20,40],[1,0]),0),colored(endpoint(1,2,[40,40],[-1,0]),0),colored(endpoint(2,3,[40,42],[-1,0]),0)]
        self.assertFalse(any(e['state']=='fragment_pair_candidate' for e in match(ends)))
    def test_no_color_abstains(self):
        self.assertIsNone(geometry(endpoint(0,1,[20,40],[1,0]),endpoint(1,2,[40,40],[-1,0])))


if __name__=='__main__':unittest.main()
