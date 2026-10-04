"""Generic invariants for the separate ROI replay, without private fixtures."""
import copy
import unittest
from audit_two_vote_resolution import prescreen


class ReplayTests(unittest.TestCase):
    def case(self):
        return dict(remaining_budget=2, seeds=[
            dict(class_id=0, box_xyxy=[20., 20., 60., 60.], pose_parent_seed_id=0,
                 semantic_model_vote_sha256=['a', 'b'], localization_voter_best_IoU={'a': .6, 'b': .6}),
            dict(class_id=0, box_xyxy=[22., 22., 62., 62.], pose_parent_seed_id=0,
                 semantic_model_vote_sha256=['a', 'b'], localization_voter_best_IoU={'a': .8, 'b': .8})])

    def test_geometry_before_semantic_probability_without_mutation(self):
        case = self.case()
        snapshot = copy.deepcopy(case)
        selected = prescreen(case, [[.001, .998, .001], [.01, .985, .005]])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['box_xyxy'], case['seeds'][1]['box_xyxy'])
        self.assertEqual(case, snapshot)

    def test_low_probability_or_wrong_class_never_launches_roi(self):
        self.assertEqual(prescreen(self.case(), [[.015, .975, .01], [.001, .001, .998]]), [])

    def test_no_budget_no_roi(self):
        case = self.case()
        case['remaining_budget'] = 0
        self.assertEqual(prescreen(case, [[.001, .998, .001]] * 2), [])

    def test_same_weight_two_views_are_not_two_voters(self):
        case = self.case()
        case['seeds'][0]['semantic_model_vote_sha256'] = ['a', 'a']
        with self.assertRaises(AssertionError):
            prescreen(case, [[.001, .998, .001]] * 2)

    def test_score_row_count_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError):
            prescreen(self.case(), [[.001, .998, .001]])


if __name__ == '__main__':
    unittest.main()
