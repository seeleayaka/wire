import unittest
import numpy as np
from segment_evidence_groups import group_segment_evidence


def record(name, ends=None, **changes):
    row = dict(record_id=name, source_score=.8, geometry_pair_eligible=True,
               crop_boundary_guard_eligible=True, candidate_tip_count=2,
               branch_cluster_count=0, skeleton_component_count=1,
               visible_ends_xy=ends or [[2, 5], [17, 5]])
    row.update(changes)
    return row


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((20, 20), dtype=bool)
        self.mask[5, 2:18] = True

    def test_exact_duplicate_reversed_ends(self):
        rows = [record('a'), record('b', [[17, 5], [2, 5]], source_score=.9)]
        result = group_segment_evidence(rows, {'a': self.mask, 'b': self.mask})
        self.assertEqual(result['review_group_count'], 1)
        self.assertEqual(result['groups'][0]['representative_original_score'], .9)
        self.assertFalse(result['automatic_connections_emitted'])
        self.assertEqual(result, group_segment_evidence(rows[::-1], {'a': self.mask, 'b': self.mask}))

    def test_adjacent_wire_is_not_duplicate(self):
        other = np.roll(self.mask, 1, axis=0)
        result = group_segment_evidence([record('a'), record('b', [[2, 6], [17, 6]])], {'a': self.mask, 'b': other})
        self.assertEqual(result['review_group_count'], 2)

    def test_boundary_evidence_not_merged(self):
        result = group_segment_evidence([record('a'), record('b', crop_boundary_guard_eligible=False)], {'a': self.mask, 'b': self.mask})
        self.assertEqual(result['review_group_count'], 2)

    def test_endpoint_disagreement_not_merged(self):
        result = group_segment_evidence([record('a'), record('b', [[2, 8], [17, 8]])], {'a': self.mask, 'b': self.mask})
        self.assertEqual(result['review_group_count'], 2)

    def test_complete_link_no_transitive_chain(self):
        mask = np.ones((20, 20), dtype=bool)
        rows = [record('a', [[2, 5], [17, 5]]), record('b', [[2, 6], [17, 6]]), record('c', [[2, 7], [17, 7]])]
        result = group_segment_evidence(rows, {r['record_id']: mask for r in rows})
        self.assertEqual([g['member_record_ids'] for g in result['groups']], [['a', 'b'], ['c']])

    def test_singleton_invalid_endpoints_rejected(self):
        for ends in ([[2, float('nan')], [17, 5]], [[-1, 5], [17, 5]], [[2, 5]]):
            with self.assertRaises(ValueError):
                group_segment_evidence([record('a', ends)], {'a': self.mask})

    def test_invalid_masks_ids_scores(self):
        for rows, masks in [([record('a'), record('a')], {'a': self.mask}),
                            ([record('a', source_score=float('nan'))], {'a': self.mask}),
                            ([record('a')], {'a': np.zeros((20, 20))}),
                            ([record('a')], {'a': self.mask, 'b': self.mask})]:
            with self.assertRaises(ValueError):
                group_segment_evidence(rows, masks)


if __name__ == '__main__':
    unittest.main()
