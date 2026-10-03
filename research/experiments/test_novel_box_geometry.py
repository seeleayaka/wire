import sys
from pathlib import Path
sys.path.insert(0,str(Path('E:/PythonProject10')))
import unittest
from novel_box_geometry import intersection_over_smaller,is_duplicate,select


class NovelGeometryTests(unittest.TestCase):
    def test_nested_boxes_not_IoU_equivalent(self):
        self.assertEqual(intersection_over_smaller([0,0,100,100],[40,40,60,60]),1.)
        self.assertTrue(is_duplicate([40,40,60,60],[{'box_xyxy':[0,0,100,100]}]))

    def test_separate_and_touching_objects(self):
        for b in ([100,0,150,50],[101,0,151,50]):
            self.assertEqual(intersection_over_smaller([0,0,100,100],b),0.)

    def test_symmetry_translation_anisotropic_scale(self):
        a=[1,3,13,21];b=[8,7,25,19];value=intersection_over_smaller(a,b)
        self.assertEqual(value,intersection_over_smaller(b,a))
        for sx,sy,dx,dy in ((3,5,21,-30),(.25,.5,-9,10)):
            f=lambda p:[p[0]*sx+dx,p[1]*sy+dy,p[2]*sx+dx,p[3]*sy+dy]
            self.assertAlmostEqual(value,intersection_over_smaller(f(a),f(b)))

    def test_invalid_fail_closed(self):
        for a in ([0,0,0,1],[0,0,float('nan'),1],[0,0,1]):
            with self.assertRaises(ValueError):intersection_over_smaller(a,[0,0,1,1])

    def test_inputs_immutable_class_agnostic(self):
        rows=[{'box_xyxy':[0,0,100,100],'class_id':0}];box=[40,40,60,60]
        self.assertTrue(is_duplicate(box,rows));self.assertEqual(box,[40,40,60,60]);self.assertEqual(rows[0]['class_id'],0)

    def candidate(self,box,parent=0,cls=0):
        return dict(box_xyxy=box,class_id=cls,confidence=.8,pose_parent_seed_id=parent,semantic_model_vote_sha256=['a','b','c'])

    def test_selector_protected_prefix_and_nested_exclusion(self):
        old=self.candidate([0,0,100,100]);current={'primary':[old],'all_predictions':[old]}
        ps=[self.candidate([20,20,40,40],0),self.candidate([150,150,250,250],1),self.candidate([175,175,200,200],2),self.candidate([300,300,400,400],3)]
        out=select(current,ps,[[.001,.998,.001]]*4,'head')
        self.assertEqual(out['all_predictions'][:1],current['all_predictions'])
        self.assertEqual(len(out['paired_semantic_additions']),2)
        self.assertEqual(len(current['all_predictions']),1)
        self.assertFalse(out['paired_semantic_additions'][0]['automatic_fault_verdict'])

    def test_selector_invalid_probabilities_not_hidden_by_duplicate(self):
        old=self.candidate([0,0,100,100]);current={'primary':[old],'all_predictions':[old]}
        with self.assertRaises(ValueError):select(current,[self.candidate([20,20,40,40])],[[0.,2.,0.]],'head')

    def test_shared_budget_not_expanded(self):
        primary=[self.candidate([i*100,0,i*100+50,50],i) for i in range(5)]
        extra=[self.candidate([i*100,100,i*100+50,150],i+5) for i in range(5)]
        current={'primary':primary,'all_predictions':primary+extra}
        out=select(current,[self.candidate([600,600,650,650])],[[.001,.998,.001]],'head')
        self.assertEqual(out['all_predictions'],current['all_predictions']);self.assertEqual(out['paired_semantic_additions'],[])


if __name__=='__main__':unittest.main()
