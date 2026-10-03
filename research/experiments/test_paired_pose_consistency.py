import unittest
from paired_pose_consistency import select


class PoseConsistency(unittest.TestCase):
    def test_isolated_high_score_is_rejected(self):
        row = dict(class_id=0, confidence=.2, box_xyxy=[100,100,150,140], semantic_model_vote_sha256=['a','b'], pose_parent_seed_id=0)
        output = select(dict(primary=[],all_predictions=[]), [row], [[.001,.998,.001]], 'h')
        self.assertEqual(output['all_predictions'], [])

    def test_identical_pixels_cannot_vote_twice(self):
        a = dict(class_id=0, confidence=.2, box_xyxy=[100,100,150,140], semantic_model_vote_sha256=['a','b'], pose_parent_seed_id=0)
        b = {**a, 'box_xyxy':[100,110,150,130]}
        self.assertEqual(select(dict(primary=[],all_predictions=[]), [a,b], [[.001,.998,.001]]*2, 'h')['all_predictions'], [])

    def test_distinct_context_consensus_still_one_alert(self):
        a = dict(class_id=0, confidence=.2, box_xyxy=[100,100,150,140], semantic_model_vote_sha256=['a','b'], pose_parent_seed_id=0)
        b = {**a, 'box_xyxy':[102,100,152,140]}
        output = select(dict(primary=[],all_predictions=[]), [a,b], [[.001,.998,.001]]*2, 'h')
        self.assertEqual(len(output['all_predictions']), 1)


if __name__ == '__main__': unittest.main()
