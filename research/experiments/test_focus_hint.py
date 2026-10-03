import copy
import unittest
import numpy as np
from inspection_agent.focus_hint import select_focus_hint


class HintTests(unittest.TestCase):
    def parent(self):return dict(left=0,top=0,right=280,bottom=210)
    def member(self,tile='tile_1',edge=False):
        return dict(left=30,top=30,right=90,bottom=90,source='tile',source_tiles=[tile],touches_tile_edge=edge)
    def test_one_hint_and_no_mutation(self):
        parent=self.parent();members=[self.member(),self.member('refine_1')];before=copy.deepcopy((parent,members))
        hint=select_focus_hint(parent,members,np.full((21,28),5.),280,210,3.5)
        self.assertEqual(hint['support_count'],1)
        self.assertFalse(hint['automatic_fault_verdict'])
        self.assertEqual((parent,members),before)
        hint['box']['left']=1
        self.assertEqual(members[0]['left'],30)
    def test_no_duplicate_tile_support(self):
        self.assertIsNone(select_focus_hint(self.parent(),[self.member(),self.member()],np.ones((21,28))*5,280,210,3.5))
    def test_threshold_is_strict(self):
        self.assertIsNone(select_focus_hint(self.parent(),[self.member(),self.member('tile_2')],np.ones((21,28))*3.5,280,210,3.5))
    def test_edges_and_large_children_excluded(self):
        members=[self.member(edge=True),self.member('tile_2',True)]
        self.assertIsNone(select_focus_hint(self.parent(),members,np.ones((21,28))*5,280,210,3.5))
        members=[dict(self.parent(),source='tile',source_tiles=[t]) for t in ('tile_1','tile_2')]
        self.assertIsNone(select_focus_hint(self.parent(),members,np.ones((21,28))*5,280,210,3.5))
    def test_bad_geometry_or_map_rejected(self):
        for score in (np.zeros((2,2)),np.full((21,28),np.nan)):
            with self.assertRaises(ValueError):select_focus_hint(self.parent(),[],score,280,210,3.5)
        child=self.member();child['left']=-1
        with self.assertRaises(ValueError):select_focus_hint(self.parent(),[child],np.zeros((21,28)),280,210,3.5)
    def test_empty_pool_abstains(self):
        self.assertIsNone(select_focus_hint(self.parent(),[],np.zeros((21,28)),280,210,3.5))
    def test_fine_grid_requires_explicit_geometry(self):
        members=[self.member(),self.member('tile_2')];score=np.ones((42,56))*5
        with self.assertRaises(ValueError):select_focus_hint(self.parent(),members,score,280,210,3.5)
        hint=select_focus_hint(self.parent(),members,score,280,210,3.5,expected_grid=(42,56))
        self.assertIsNotNone(hint)


if __name__=='__main__':unittest.main()
