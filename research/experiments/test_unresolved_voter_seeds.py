import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path('E:/PythonProject10')))
import unittest
from unresolved_voter_seeds import seeds,windows


class UnresolvedTests(unittest.TestCase):
    def model(self,weight,box=[100,100,200,200]):
        return dict(weight_sha256=weight,source_sha256='source',predictions=dict(source_shape=[1200,1600],merged_predictions=[dict(box_xyxy=list(box),class_id=0,confidence=.8)]))

    def test_two_shas_only_seeds_not_cues(self):
        result=seeds([self.model('a'),self.model('b')],{'all_predictions':[]},'source',(1200,1600))
        self.assertTrue(result)
        self.assertTrue(all(r['seed_only_requires_third_checkpoint'] and not r['automatic_fault_verdict'] for r in result))
        self.assertTrue(all(len(r['semantic_model_vote_sha256'])==2 for r in result))

    def test_repeated_views_not_three_votes(self):
        result=seeds([self.model('a'),self.model('a'),self.model('b')],{'all_predictions':[]},'source',(1200,1600))
        self.assertTrue(result)
        self.assertFalse(seeds([self.model('a')]*3,{'all_predictions':[]},'source',(1200,1600)))
        self.assertFalse(seeds([self.model(w) for w in ('a','b','c')],{'all_predictions':[]},'source',(1200,1600)))

    def test_existing_nested_objects_excluded(self):
        self.assertFalse(seeds([self.model('a'),self.model('b')],{'all_predictions':[dict(box_xyxy=[50,50,250,250])]},'source',(1200,1600)))

    def test_model_order_deterministic_input_immutable(self):
        models=[self.model('a'),self.model('b')];before=copy.deepcopy(models)
        a=seeds(models,{'all_predictions':[]},'source',(1200,1600));b=seeds(models[::-1],{'all_predictions':[]},'source',(1200,1600))
        self.assertEqual(a,b);self.assertEqual(models,before)

    def test_wrong_source_and_frame_rejected(self):
        with self.assertRaises(ValueError):seeds([self.model('a')],{'all_predictions':[]},'wrong',(1200,1600))
        with self.assertRaises(ValueError):seeds([self.model('a')],{'all_predictions':[]},'source',(1200,1601))

    def test_windows_geometry_bounded_and_context_pair(self):
        for box in ([30,30,60,60],[1400,1000,1500,1100],[700,500,800,600]):
            ws=windows(dict(box_xyxy=box),(1200,1600));self.assertEqual(len(ws),2)
            for l,t,r,b in ws:self.assertTrue(0<=l<r<=1600 and 0<=t<b<=1200);self.assertEqual((r-l,b-t),(960,960))
        with self.assertRaises(ValueError):windows(dict(box_xyxy=[1,1,10,10]),(500,500))


if __name__=='__main__':unittest.main()
