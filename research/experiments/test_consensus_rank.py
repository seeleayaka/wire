import sys
from pathlib import Path
sys.path.insert(0,str(Path('E:/PythonProject10')))
import unittest
from consensus_rank import select,validate_geometry_score


class ConsensusRankTests(unittest.TestCase):
    def row(self,box,parent,quality):
        return dict(box_xyxy=box,confidence=.8,class_id=0,pose_parent_seed_id=parent,
            semantic_model_vote_sha256=['a','b','c'],localization_voter_best_IoU=dict(a=quality,b=quality,c=quality))

    def test_geometry_first_without_lowering_probability(self):
        current={'primary':[],'all_predictions':[]}
        a=self.row([100,100,140,140],0,.51);b=self.row([90,90,150,150],0,.8)
        out=select(current,[a,b],[[.001,.998,.001],[.009,.982,.009]],'head')
        self.assertEqual(out['paired_semantic_additions'][0]['box_xyxy'],b['box_xyxy'])
        out=select(current,[a,b],[[.001,.998,.001],[.015,.97,.015]],'head')
        self.assertEqual(out['paired_semantic_additions'][0]['box_xyxy'],a['box_xyxy'])

    def test_geometry_vote_mask_immutable(self):
        row=self.row([1,1,30,30],0,.7);before=row.copy()
        self.assertAlmostEqual(validate_geometry_score(row),.7);self.assertEqual(row,before)

    def test_missing_nonfinite_and_nonvoting_geometry_rejected(self):
        for values in ({'a':.7,'b':.7},{'a':.7,'b':.7,'c':float('nan')},{'a':.7,'b':.7,'c':.4}):
            row=self.row([1,1,30,30],0,.7);row['localization_voter_best_IoU']=values
            with self.assertRaises(ValueError):validate_geometry_score(row)

    def test_parent_and_candidate_order_deterministic(self):
        current={'primary':[],'all_predictions':[]};a=self.row([100,100,140,140],0,.8);b=self.row([90,90,150,150],0,.8)
        x=select(current,[a,b],[[.005,.99,.005]]*2,'head');y=select(current,[b,a],[[.005,.99,.005]]*2,'head')
        self.assertEqual(x,y)

    def test_prefix_and_cross_class_nested_not_overwritten(self):
        old=dict(box_xyxy=[0,0,100,100],class_id=1,confidence=.8)
        current={'primary':[old],'all_predictions':[old]};r=self.row([20,20,40,40],0,.9)
        out=select(current,[r],[[.001,.998,.001]],'head')
        self.assertEqual(out['all_predictions'],[old]);self.assertEqual(current['all_predictions'],[old])


if __name__=='__main__':unittest.main()
