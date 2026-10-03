import unittest
from tools.probe_cnn_candidate_rank import box_union_area


class UnionTests(unittest.TestCase):
    def box(self,l,t,r,b): return dict(left=l,top=t,right=r,bottom=b)
    def test_union_ignores_overlap_duplicates_and_nested_boxes(self):
        a=self.box(0,0,2,2); b=self.box(1,1,3,3)
        self.assertEqual(box_union_area([a,b]),7)
        self.assertEqual(box_union_area([a,a,self.box(.5,.5,1,1)]),4)
        self.assertEqual(box_union_area([]),0)
    def test_disjoint_and_invalid_geometry(self):
        self.assertEqual(box_union_area([self.box(0,0,1,1),self.box(2,0,3,2)]),3)
        with self.assertRaises(ValueError): box_union_area([self.box(2,0,1,1)])
        with self.assertRaises(ValueError): box_union_area([self.box(0,0,float('nan'),1)])


if __name__=='__main__': unittest.main()
