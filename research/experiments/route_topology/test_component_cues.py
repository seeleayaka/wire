import unittest
import cv2
import numpy as np
from component_cues import color_components,color_descriptor,semantic_descriptor,fit_head,prediction,gate

class ComponentTests(unittest.TestCase):
    def test_colour_partition_covers_all_actual_pixels(self):
        x=np.random.default_rng(5).integers(0,256,(50,100,3),dtype=np.uint8)
        labels=color_components(x);self.assertTrue(np.isin(labels,np.arange(8)).all())
        f=color_descriptor(x);self.assertEqual(f.shape,(128,));self.assertTrue(np.isfinite(f).all())
        np.testing.assert_allclose(f[:64].reshape(8,8).sum(1),1.)
    def test_neutral_does_not_become_coloured_wire(self):
        for v,expected in [(0,5),(100,7),(255,6)]:
            self.assertTrue((color_components(np.full((50,100,3),v,np.uint8))==expected).all())
    def test_red_blue_and_yellow_separate(self):
        for rgb,k in [([255,0,0],0),([255,255,0],1),([0,0,255],3)]:
            self.assertTrue((color_components(np.tile(np.uint8(rgb),(50,100,1)))==k).all())
    def test_gap_is_not_closed(self):
        x=np.zeros((50,100,3),np.uint8);x[20:30,10:90]=[255,0,0];x[:,49:51]=0
        mask=(color_components(x)==0).astype(np.uint8)
        self.assertEqual(cv2.connectedComponents(mask,connectivity=8)[0]-1,2)
    def test_shape_rejects(self):
        with self.assertRaises(ValueError):color_descriptor(np.zeros((2,2,3),np.uint8))
    def test_component_cluster_label_permutation_keeps_matching_geometry(self):
        rng=np.random.default_rng(8);centers=rng.normal(size=(12,4));tokens=centers[np.arange(128)%12]
        a,labels=semantic_descriptor(tokens,centers);self.assertEqual(a.shape,(72,))
        np.testing.assert_allclose(a[:60].reshape(5,12).sum(1),1.)
        perm=rng.permutation(12);b,_=semantic_descriptor(tokens,centers[perm])
        np.testing.assert_allclose(a[:60].reshape(5,12)[:,perm],b[:60].reshape(5,12))
    def test_fit_cannot_use_too_few_originals(self):
        with self.assertRaises(ValueError):fit_head(np.ones((4,10)),[0,0,1,1])
    def test_leave_self_out_removes_evaluated_score(self):
        model=dict(weights=np.zeros(1),center=np.zeros(1),scale=np.ones(1),bias=0.)
        cal={k:[dict(id=f'{k}-{i}',score=.5) for i in range(20)] for k in [0,1]}
        r=prediction(model,[0],cal,'0-0');self.assertTrue(r['leave_self_out'])
        self.assertIsNone(r['visual_label_candidate']);self.assertEqual(r['new_confirmed_connections'],0)
    def test_wrong_singleton_fails_gate(self):
        rows=[dict(visual_label=0,prediction=dict(visual_label_candidate=1))]
        self.assertFalse(gate(rows)['passed'])
if __name__=='__main__':unittest.main()
