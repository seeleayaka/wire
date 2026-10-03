import copy
import unittest
from paired_pose_search import proposals, select


class PoseSearch(unittest.TestCase):
    def test_same_parent_gets_one_alert_even_when_scales_do_not_overlap50(self):
        base = dict(primary=[], all_predictions=[])
        a = dict(class_id=0, confidence=.1, box_xyxy=[80, 80, 120, 120], semantic_model_vote_sha256=['a', 'b'], pose_parent_seed_id=0)
        b = {**a, 'box_xyxy': [70, 70, 130, 130]}
        before = copy.deepcopy([a, b])
        result = select(base, [a, b], [[.001, .998, .001], [.0005, .999, .0005]], 'same_head')
        self.assertEqual(len(result['paired_semantic_additions']), 1)
        self.assertEqual(result['paired_semantic_additions'][0]['box_xyxy'], b['box_xyxy'])
        self.assertEqual([a, b], before)

    def test_every_changed_pose_rechecks_two_distinct_votes(self):
        def model(weight):
            return dict(source_sha256='s', weight_sha256=weight, predictions=dict(source_shape=[300, 300],
                merged_predictions=[dict(class_id=0, confidence=.1, box_xyxy=[100, 100, 160, 140])]))
        current = dict(primary=[], all_predictions=[])
        self.assertEqual(proposals(model('a'), [model('a'), model('a')], current), [])
        values = proposals(model('a'), [model('a'), model('b')], current)
        self.assertEqual(len(values), 7)
        self.assertTrue(all(len(set(row['semantic_model_vote_sha256'])) == 2 for row in values))

    def test_old_prefix_and_fixed_probability_are_retained(self):
        old = dict(class_id=0, confidence=.8, box_xyxy=[200, 200, 220, 220])
        current = dict(primary=[old], all_predictions=[old])
        row = dict(class_id=0, confidence=.1, box_xyxy=[100, 100, 140, 140], semantic_model_vote_sha256=['a', 'b'], pose_parent_seed_id=0)
        self.assertFalse(select(current, [row], [[.011, .979, .01]], 'same')['paired_semantic_additions'])
        output = select(current, [row], [[.01, .98, .01]], 'same')
        self.assertEqual(output['all_predictions'][0], old)


if __name__ == '__main__': unittest.main()
