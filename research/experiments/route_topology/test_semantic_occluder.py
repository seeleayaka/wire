import unittest
from run_semantic_occluder import semantic_edges
import numpy as np


class SemanticSelection(unittest.TestCase):
    def setUp(self):
        self.raw=np.zeros((30,40),bool);self.raw[15,3:12]=True;self.raw[15,20:30]=True
        self.ends=[dict(endpoint_id=0,point_xy=[11,15],width=4),dict(endpoint_id=1,point_xy=[20,15],width=4)]
        self.edges=[dict(endpoint_ids=[0,1],score=.3,state='unaccepted',virtual_curve_xy=[[x,15] for x in range(11,21)])]
        mask=np.zeros_like(self.raw);mask[10:21,13:19]=True;yy,xx=np.nonzero(mask)
        self.obj=dict(kind='label',component=1,pixels=int(mask.sum()),axis_ratio=None,mask=mask,xy=np.column_stack((xx,yy)))

    def test_no_object_no_join(self):
        self.assertNotEqual(semantic_edges(self.ends,self.edges,self.raw,[])[0]['state'],'occluder_supported_fragment_candidate')

    def test_one_object_inferred_only(self):
        edge=semantic_edges(self.ends,self.edges,self.raw,[self.obj])[0]
        self.assertEqual(edge['state'],'occluder_supported_fragment_candidate')
        self.assertFalse(edge['physical_identity_confirmed']);self.assertEqual(edge['observed_gap_pixels'],0)
        self.assertEqual(self.edges[0]['state'],'unaccepted')

    def test_ambiguous_objects_no_join(self):
        self.assertNotEqual(semantic_edges(self.ends,self.edges,self.raw,[self.obj,self.obj])[0]['state'],'occluder_supported_fragment_candidate')


if __name__=='__main__':unittest.main()
