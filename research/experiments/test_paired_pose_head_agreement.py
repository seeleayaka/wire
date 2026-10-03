import unittest
from paired_pose_head_agreement import select


class PoseHeadAgreement(unittest.TestCase):
    def row(self):
        return dict(class_id=0,confidence=.1,box_xyxy=[100,100,150,140],semantic_model_vote_sha256=['a','b'],pose_parent_seed_id=0)
    def test_same_box_both_heads_required(self):
        base = dict(primary=[],all_predictions=[])
        row = self.row()
        self.assertEqual(select(base,[row],[[.001,.998,.001]],[[.3,.699,.001]],'old','new')['all_predictions'], [])
        result = select(base,[row],[[.001,.998,.001]],[[.002,.997,.001]],'old','new')
        self.assertEqual(len(result['all_predictions']),1)
        self.assertEqual(result['all_predictions'][0]['confidence'],.998)
        self.assertEqual(result['all_predictions'][0]['pose_auxiliary_probability'],.997)
    def test_auxiliary_probability_validation_not_bypassed(self):
        with self.assertRaises(ValueError):
            select(dict(primary=[],all_predictions=[]),[self.row()],[[1,0,0]],[[float('nan'),0,0]],'old','new')


if __name__ == '__main__': unittest.main()
