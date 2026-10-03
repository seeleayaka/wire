import copy
import unittest

from test_teacher_student_port_policy import case, row
from port_residual_feature_support import merge_residual


def fixture():
    teacher = case([row(100 + i * 60, .99 - i * .005) for i in range(5)]
                   + [row(1200, .3), row(1500, .3)])
    old = case([row(1200)])
    old['predictions']['edge_kept_predictions'].append(row(1200, .85, tile=1))
    new = case([row(1500)])
    new['predictions']['edge_kept_predictions'].append(row(1500, .85, tile=1))
    return teacher, old, new


class ResidualTests(unittest.TestCase):
    def test_preserves_accepted_prefix_and_inputs(self):
        from teacher_student_port_policy import merge
        teacher, old, new = fixture()
        before = copy.deepcopy((teacher, old, new))
        accepted = merge(teacher, old)
        result = merge_residual(teacher, old, new)
        self.assertEqual(result['all_predictions'][:len(accepted['all_predictions'])], accepted['all_predictions'])
        self.assertEqual(len(result['feature_additions']), 1)
        self.assertEqual((teacher, old, new), before)

    def test_new_model_losing_old_detection_does_not_remove_it(self):
        teacher, old, new = fixture()
        result = merge_residual(teacher, old, new)
        self.assertTrue(any(p['box_xyxy'][0] == 1200 for p in result['all_predictions']))

    def test_shared_extra_budget(self):
        teacher, old, new = fixture()
        rows = [row(100 + i * 60, .99 - i * .005) for i in range(10)]
        teacher['predictions']['merged_predictions'] = rows + [row(1500, .3)]
        teacher['predictions']['edge_kept_predictions'] = rows + [dict(r, source_tile=1) for r in rows]
        result = merge_residual(teacher, old, new)
        self.assertEqual(len(result['all_predictions']), 10)
        self.assertEqual(result['feature_additions'], [])

    def test_deduplicate_against_old_student(self):
        teacher, old, _ = fixture()
        self.assertEqual(merge_residual(teacher, old, old)['feature_additions'], [])

    def test_identity_failure_preserves_accepted(self):
        teacher, old, new = fixture()
        new['source_sha256'] = 'different'
        result = merge_residual(teacher, old, new)
        self.assertEqual(result['feature_additions'], [])
        self.assertEqual(len(result['all_predictions']), 6)
        self.assertIsNotNone(result['feature_fallback_reason'])

    def test_no_teacher_support_no_new_cue(self):
        teacher, old, new = fixture()
        teacher['predictions']['merged_predictions'][-1]['confidence'] = .25
        self.assertEqual(merge_residual(teacher, old, new)['feature_additions'], [])

    def test_two_distinct_views_required(self):
        teacher, old, new = fixture()
        new['predictions']['edge_kept_predictions'][1]['source_tile'] = 0
        self.assertEqual(merge_residual(teacher, old, new)['feature_additions'], [])

    def test_boundary_not_relaxed(self):
        teacher, old, new = fixture()
        new['predictions']['merged_predictions'][0]['confidence'] = .75
        self.assertEqual(merge_residual(teacher, old, new)['feature_additions'], [])


if __name__ == '__main__':
    unittest.main()
