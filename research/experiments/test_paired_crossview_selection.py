import copy,unittest,sys
sys.path.insert(0,'E:/PythonProject10')
from paired_crossview_selection import select


class CrossviewTests(unittest.TestCase):
    def setUp(self):
        self.current=dict(primary=[],all_predictions=[])
        self.proposal=dict(box_xyxy=[100,100,150,150],class_id=0,confidence=.1,semantic_model_vote_sha256=['t','s'])

    def test_two_views_min_score_and_raw_evidence(self):
        original=copy.deepcopy((self.current,self.proposal))
        result=select(self.current,[self.proposal],[[[.01,.98,.01]],[[.005,.99,.005]]],['context','footprint'])
        self.assertEqual(len(result['paired_crossview_additions']),1);self.assertEqual(result['all_predictions'][0]['confidence'],.98)
        self.assertEqual((self.current,self.proposal),original)
        self.assertIn('not_calibrated',result['all_predictions'][0]['score_kind'])

    def test_weak_or_conflicting_views_and_old_overlap_reject(self):
        for other in ([.02,.97,.01],[.01,.01,.98]):
            self.assertFalse(select(self.current,[self.proposal],[[[.01,.98,.01]],[other]],['c','f'])['paired_crossview_additions'])
        self.current['primary']=[self.proposal];self.current['all_predictions']=[self.proposal]
        self.assertEqual(select(self.current,[self.proposal],[[[.01,.98,.01]],[[.01,.98,.01]]],['c','f'])['all_predictions'],[self.proposal])
        with self.assertRaises(ValueError):select(self.current,[self.proposal],[[[.01,.98,.01]],[[.01,.98,.01]]],['same','same'])

if __name__=='__main__':unittest.main()
