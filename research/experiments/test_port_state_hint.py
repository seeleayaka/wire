from copy import deepcopy
import unittest
from inspection_agent.port_state_hint import select_port_state_hints


def prediction(**extra):
    return dict(left=10,top=10,right=20,bottom=20,confidence=.8,
                valid_warp_fraction=1.,class_id=0,**extra)


class PortHintTests(unittest.TestCase):
    def setUp(self): self.parents=[dict(left=0,top=0,right=100,bottom=100,source='frozen')]
    def hints(self, preds, parents=None, threshold=.5):
        return select_port_state_hints(self.parents if parents is None else parents,preds,threshold)
    def test_parent_geometry_order_and_metadata_are_unchanged(self):
        before=deepcopy(self.parents); out=self.hints([prediction()])
        self.assertEqual(out['parents'],before); self.assertEqual(self.parents,before)
        self.assertEqual(len(out['hints']),1); self.assertFalse(out['hints'][0]['automatic_fault_verdict'])
        out['parents'][0]['left']=9; self.assertEqual(self.parents,before)
    def test_threshold_boundary_and_invalid_warp_abstain(self):
        for key,value in (('confidence',.5),('valid_warp_fraction',.979)):
            p=prediction();p[key]=value; self.assertEqual(self.hints([p])['hints'],[])
    def test_outside_or_too_large_box_abstains(self):
        for bounds in ((90,90,110,110),(0,0,100,80)):
            p=prediction();p.update(zip(('left','top','right','bottom'),bounds))
            self.assertEqual(self.hints([p])['hints'],[])
    def test_smallest_parent_assignment_is_unique(self):
        parents=self.parents+[dict(left=5,top=5,right=25,bottom=25)]
        out=self.hints([prediction()],parents)
        self.assertEqual([h['parent_index'] for h in out['hints']],[1])
    def test_maximum_one_hint_and_stable_order(self):
        a,b=prediction(),prediction(); b.update(left=30,right=40,confidence=.9)
        for preds in ([a,b],[b,a]):
            out=self.hints(preds);self.assertEqual(len(out['hints']),1)
            self.assertEqual(out['hints'][0]['box'],b)
    def test_duplicates_do_not_consume_two_parents(self):
        out=self.hints([prediction(),prediction()],self.parents*2)
        self.assertEqual(len(out['hints']),1)
    def test_duplicate_geometry_keeps_best_score_in_either_order(self):
        a,b=prediction(),prediction();b['confidence']=.95
        for order in ([a,b],[b,a]):
            self.assertEqual(self.hints(order)['hints'][0]['box']['confidence'],.95)
    def test_invalid_scores_geometry_and_classes_abstain(self):
        for field,value in (('confidence',float('nan')),('confidence',1.1),('class_id',2),
                            ('class_id',True),('left',-1),('right',5),('valid_warp_fraction',float('inf'))):
            p=prediction();p[field]=value;self.assertEqual(self.hints([p])['hints'],[])
    def test_invalid_parent_or_threshold_rejected(self):
        for threshold in (.2,1.1,float('nan'),True):
            with self.assertRaises(ValueError):self.hints([],threshold=threshold)
        with self.assertRaises(ValueError):self.hints([],parents=[{'left':0}])


if __name__=='__main__':unittest.main()
