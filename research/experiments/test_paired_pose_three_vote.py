import unittest
from paired_pose_three_vote import select


class ThreeVotes(unittest.TestCase):
    def test_two_checkpoints_and_repeated_views_do_not_pass(self):
        row = dict(class_id=0,confidence=.1,box_xyxy=[100,100,150,140],semantic_model_vote_sha256=['a','b','a'],pose_parent_seed_id=0)
        base = dict(primary=[],all_predictions=[])
        self.assertEqual(select(base,[row],[[.001,.998,.001]],'h')['all_predictions'],[])
        row['semantic_model_vote_sha256'] = ['a','b','c']
        self.assertEqual(len(select(base,[row],[[.001,.998,.001]],'h')['all_predictions']),1)


if __name__ == '__main__': unittest.main()
