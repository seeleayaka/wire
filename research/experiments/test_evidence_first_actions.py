import copy
import unittest
from audit_evidence_first_resolution import geometric_actions


class ActionTests(unittest.TestCase):
    def seed(self, parent=0, box=None, value=.8):
        return dict(class_id=0, box_xyxy=box or [20., 20., 60., 60.], pose_parent_seed_id=parent,
                    semantic_model_vote_sha256=['a', 'b'], localization_voter_best_IoU={'a': value, 'b': value})

    def test_action_needs_no_fault_probability_and_preserves_input(self):
        case = dict(remaining_budget=1, seeds=[self.seed()])
        before = copy.deepcopy(case)
        self.assertEqual(len(geometric_actions(case, [])), 1)
        self.assertEqual(case, before)
        with self.assertRaises(AssertionError):
            geometric_actions(case, [[0., 1., 0.]])

    def test_one_per_parent_geometry_first(self):
        case = dict(remaining_budget=5, seeds=[self.seed(value=.6), self.seed(box=[22., 22., 62., 62.])])
        self.assertEqual(geometric_actions(case, [])[0]['box_xyxy'], [22., 22., 62., 62.])

    def test_budget_and_duplicate_object_guard(self):
        case = dict(remaining_budget=1, seeds=[self.seed(), self.seed(1, [22., 22., 62., 62.]), self.seed(2, [100., 100., 140., 140.])])
        self.assertEqual(len(geometric_actions(case, [])), 1)
        case['remaining_budget'] = 5
        self.assertEqual(len(geometric_actions(case, [])), 2)

    def test_one_checkpoint_multiple_views_never_two_voters(self):
        row = self.seed()
        row['semantic_model_vote_sha256'] = ['a', 'a']
        with self.assertRaises(AssertionError):
            geometric_actions(dict(remaining_budget=1, seeds=[row]), [])

    def test_source_order_irrelevant_to_rank(self):
        case = dict(remaining_budget=1, seeds=[self.seed(), self.seed(2, [100., 100., 140., 140.], .9)])
        reversed_case = dict(case, seeds=list(reversed(case['seeds'])))
        self.assertEqual(geometric_actions(case, []), geometric_actions(reversed_case, []))


if __name__ == '__main__':
    unittest.main()
