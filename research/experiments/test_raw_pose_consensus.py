import sys
from pathlib import Path
sys.path.insert(0,'E:/PythonProject10')
import unittest
from raw_pose_consensus import proposals,select

class RawPose(unittest.TestCase):
    def model(self,weight='a'):
        return dict(source_sha256='input',weight_sha256=weight,predictions=dict(source_shape=[400,500],merged_predictions=[dict(class_id=0,confidence=.8,box_xyxy=[100,100,150,140])]))
    def test_duplicate_views_are_not_checkpoint_votes(self):
        self.assertEqual(proposals(self.model(),[self.model(),self.model(),self.model('b')],dict(primary=[],all_predictions=[])),[])
    def test_three_votes_and_shared_budget_preserve_prefix(self):
        models=[self.model(w) for w in ('a','b','c')];old=dict(primary=[],all_predictions=[])
        rows=proposals(models[0],models,old);self.assertTrue(rows)
        out=select(old,rows,[[.001,.998,.001]]*len(rows),'head')
        self.assertEqual(len(out['all_predictions']),1);self.assertEqual(old['all_predictions'],[])
    def test_no_duplicate_of_old_cue(self):
        models=[self.model(w) for w in ('a','b','c')]
        self.assertEqual(proposals(models[0],models,dict(primary=[],all_predictions=[models[0]['predictions']['merged_predictions'][0]])),[])
    def test_mismatched_source_rejected(self):
        model=self.model('b');model['source_sha256']='other'
        with self.assertRaises(ValueError):proposals(self.model(),[model],dict(primary=[],all_predictions=[]))

if __name__=='__main__':unittest.main()
