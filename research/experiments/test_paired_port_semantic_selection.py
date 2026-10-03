import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
from paired_port_semantic_selection import select


def proposal(x=1500):return dict(class_id=0,confidence=.1,box_xyxy=[x,1000,x+50,1050],semantic_model_vote_sha256=['teacher','student'])
def baseline():return dict(primary=[],all_predictions=[])


class SelectionTests(unittest.TestCase):
    def test_effective_confidence_has_explicit_low_detector_score(self):
        current=baseline();before=copy.deepcopy(current);p=proposal()
        result=select(current,[p],[[.01,.98,.01]],'head')
        self.assertEqual(len(result['paired_semantic_additions']),1)
        self.assertEqual(result['all_predictions'][0]['confidence'],.98)
        self.assertEqual(result['all_predictions'][0]['proposal_detector_score'],.1)
        self.assertEqual(current,before);self.assertEqual(p['confidence'],.1)
    def test_wrong_class_low_probability_and_bad_votes(self):
        self.assertEqual(select(baseline(),[proposal()],[[.01,.01,.98]],'head')['paired_semantic_additions'],[])
        self.assertEqual(select(baseline(),[proposal()],[[.02,.97,.01]],'head')['paired_semantic_additions'],[])
        p=proposal();p['semantic_model_vote_sha256']=['same','same']
        with self.assertRaises(ValueError):select(baseline(),[p],[[.01,.98,.01]],'head')
        with self.assertRaises(ValueError):select(baseline(),[proposal()],[[0,1,1]],'head')
    def test_old_prefix_and_full_shared_budget(self):
        old=dict(primary=[],all_predictions=[proposal(2000+i*60) for i in range(5)]);before=copy.deepcopy(old)
        self.assertEqual(select(old,[proposal()],[[.01,.98,.01]],'head')['paired_semantic_additions'],[])
        self.assertEqual(old,before)


if __name__=='__main__':unittest.main()
