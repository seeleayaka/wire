import unittest
from copy import deepcopy
from unittest.mock import patch
from port_segment_review import review_port_contacts

BIND = dict(image_sha256='fixture', image_size=[100, 100], coordinate_frame='source_image_pixels')


def port(name='p', box=None, confirmed=True):
    return dict(id=name, bbox_xyxy=box or [0, 0, 10, 10], confirmed=confirmed)


def record(name='r', ends=None, **changes):
    result = dict(record_id=name, visible_ends_xy=ends or [[5, 5], [50, 50]], source_score=.9,
                  geometry_pair_eligible=True, candidate_tip_count=2, branch_cluster_count=0,
                  skeleton_component_count=1)
    result.update(changes)
    return result


class ContactTests(unittest.TestCase):
    def run_review(self, ports=None, records=None, binding=None):
        with patch('port_segment_review.validate_mapping', return_value=BIND):
            return review_port_contacts(dict(map_id='fixture', ports=ports if ports is not None else [port()]), 'fixture.png',
                dict(geometry_contract_version=1, image_binding=binding or BIND,
                     records=records if records is not None else [record()]))

    def test_unique_contact_still_not_connection(self):
        result = self.run_review()
        self.assertEqual(result['rows'][0]['state'], 'local_contact_manual_review')
        self.assertEqual(result['connection_edges'], [])
        self.assertFalse(result['rows'][0]['confirmed_assignment'])

    def test_draft_never_confirmed(self):
        self.assertEqual(self.run_review([port(confirmed=False)])['rows'][0]['state'], 'draft_roi_contact_unconfirmed')

    def test_nearest_does_not_assign(self):
        result = self.run_review(records=[record(ends=[[10, 5], [50, 50]])])
        self.assertEqual(result['rows'][0]['state'], 'no_roi_contact_not_missing_wire')
        self.assertEqual(result['rows'][0]['nearest_diagnostics_not_assignments'][0]['distance_to_roi_closure_px'], 0)

    def test_overlap_ambiguous(self):
        self.assertEqual(self.run_review([port(), port('q')])['rows'][0]['state'], 'ambiguous_overlapping_rois')

    def test_shared_port_ambiguous(self):
        result = self.run_review(records=[record(), record('s')])
        self.assertEqual(result['rows'][0]['state'], 'ambiguous_shared_port_evidence')

    def test_same_segment_both_ends_one_port_ambiguous(self):
        result = self.run_review(records=[record(ends=[[3, 5], [7, 5]])])
        self.assertTrue(all(r['state'] == 'ambiguous_shared_port_evidence' for r in result['rows']))

    def test_crop_and_branch_abstain(self):
        for changes in [dict(crop_evidence=dict(boundary_truncated=True)), dict(branch_cluster_count=1), dict(geometry_pair_eligible=False)]:
            result = self.run_review(records=[record(**changes)])
            self.assertEqual(len(result['abstained_records']), 1)
            self.assertEqual(result['rows'], [])

    def test_invalid_inputs(self):
        for rows in [[record(), record()], [record(source_score=float('nan'))],
                     [record(ends=[[100, 0], [1, 1]])], [record(ends=[[True, 0], [1, 1]])]]:
            with self.assertRaises(ValueError):
                self.run_review(records=rows)

    def test_image_binding_rejected(self):
        wrong = dict(BIND, image_sha256='other')
        with self.assertRaises(ValueError):
            self.run_review(binding=wrong)

    def test_order_invariant_and_no_mutation(self):
        rows = [record(), record('s')]
        original = deepcopy(rows)
        self.assertEqual(self.run_review(records=rows), self.run_review(records=rows[::-1]))
        self.assertEqual(rows, original)

    def test_empty_not_all_clear(self):
        self.assertEqual(self.run_review(records=[])['decision'], 'insufficient_evidence')

    def test_absent_map_distinct_from_no_contact(self):
        self.assertTrue(all(row['state'] == 'port_map_missing' for row in self.run_review(ports=[])['rows']))


if __name__ == '__main__':
    unittest.main()
