import copy
import sys
import unittest
sys.path.insert(0,'E:/PythonProject10')
from paired_semantic_committee import select_committee


class CommitteeTests(unittest.TestCase):
    def test_all_three_members_required_not_average(self):
        current=dict(primary=[],all_predictions=[])
        row=dict(class_id=0,confidence=.1,box_xyxy=[100,100,150,150],semantic_model_vote_sha256=['t','s'])
        raw=[[[.01,.98,.01]],[[.005,.99,.005]],[[.005,.995,0]]]
        before=copy.deepcopy(raw);result=select_committee(current,[row],raw,['a','b','c'])
        self.assertEqual(result['all_predictions'][0]['confidence'],.98)
        self.assertEqual(result['all_predictions'][0]['paired_committee_member_probabilities'],raw[0]+raw[1]+raw[2])
        self.assertEqual(raw,before);self.assertEqual(row['confidence'],.1)
        raw[2]=[[.03,.97,0]]
        self.assertEqual(select_committee(current,[row],raw,['a','b','c'])['paired_committee_additions'],[])

    def test_wrong_class_and_same_head_fail_closed(self):
        current=dict(primary=[],all_predictions=[])
        row=dict(class_id=0,confidence=.1,box_xyxy=[100,100,150,150],semantic_model_vote_sha256=['t','s'])
        raw=[[[.01,.98,.01]],[[.01,.98,.01]],[[.01,.01,.98]]]
        self.assertEqual(select_committee(current,[row],raw,['a','b','c'])['paired_committee_additions'],[])
        with self.assertRaises(ValueError):select_committee(current,[row],raw,['a','a','c'])


if __name__=='__main__':unittest.main()
