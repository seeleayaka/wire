from __future__ import annotations

from copy import deepcopy
import unittest
import numpy as np
import cv2

from inspection_agent.visible_segment_geometry import assess_visible_skeleton
from inspection_agent.sam_topology_adapter import connection_graph_from_sam_endpoints


class VisibleGeometryTests(unittest.TestCase):
    def test_straight_and_diagonal_open_segments(self):
        for diagonal in (False, True):
            image = np.zeros((25, 25), np.uint8)
            cv2.line(image, (3, 3), (21, 21 if diagonal else 3), 1, 1)
            result = assess_visible_skeleton(image)
            self.assertTrue(result['geometry_pair_eligible'])
            self.assertEqual(len(result['visible_ends_xy']), 2)
            self.assertEqual(result['branch_cluster_count'], 0)

    def test_y_and_t_junctions_abstain_without_pruning(self):
        for upper in ((10, 2), (2, 2)):
            image = np.zeros((25, 25), np.uint8)
            for endpoint in (upper, (22, 10), (10, 22)):
                cv2.line(image, (10, 10), endpoint, 1, 1)
            result = assess_visible_skeleton(image)
            self.assertFalse(result['geometry_pair_eligible'])
            self.assertGreater(result['branch_cluster_count'], 0)
            self.assertEqual(result['visible_ends_xy'], [])
            self.assertGreaterEqual(result['candidate_tip_count'], 3)

    def test_closed_loop_never_supplies_pair(self):
        image = np.zeros((25, 25), np.uint8)
        cv2.circle(image, (12, 12), 7, 1, 1)
        self.assertFalse(assess_visible_skeleton(image)['geometry_pair_eligible'])

    def test_empty_and_disconnected_skeleton_abstain(self):
        image = np.zeros((25, 25), np.uint8)
        self.assertFalse(assess_visible_skeleton(image)['geometry_pair_eligible'])
        image[3, 3:10] = 1
        image[12, 3:10] = 1
        result = assess_visible_skeleton(image)
        self.assertEqual(result['skeleton_component_count'], 2)
        self.assertEqual(result['visible_ends_xy'], [])

    def test_input_validation_and_no_mutation(self):
        for bad in (np.zeros((0, 2)), np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError): assess_visible_skeleton(bad)
        image = np.zeros((25, 25), np.uint8); image[3, 3:22] = 255
        before = image.copy(); result = assess_visible_skeleton(image)
        np.testing.assert_array_equal(image, before)
        result['visible_ends_xy'].append([24, 24])
        self.assertEqual(len(assess_visible_skeleton(image)['visible_ends_xy']), 2)


class AdapterGeometryTests(unittest.TestCase):
    def setUp(self):
        self.terminals = {'schema_version': 1, 'graph_id': 'controlled',
            'scene_type': 'bench_terminal_board', 'terminals': [
                {'id': 'A', 'bbox_xyxy': [0, 0, 7, 7]},
                {'id': 'B', 'bbox_xyxy': [18, 0, 24, 7]}]}
        image = np.zeros((25, 25), np.uint8); image[3, 3:22] = 1
        self.record = {'record_id': 'm1_c1', 'source_score': .9,
                       **assess_visible_skeleton(image)}

    def adapt(self, record, **meta):
        return connection_graph_from_sam_endpoints({'records': [record], **meta}, self.terminals)

    def test_verified_simple_pair_is_preserved(self):
        graph, report = self.adapt(self.record)
        self.assertEqual(len(graph.connections), 1)
        self.assertTrue(report['evidence_records'][0]['geometry_evidence_verified'])

    def test_legacy_collapsed_many_tips_never_emits_edge(self):
        record = {'record_id': 'old', 'source_score': .9,
                  'visible_ends_xy': [[3, 3], [21, 3]], 'candidate_tip_count': 5}
        graph, report = self.adapt(record)
        self.assertEqual(len(graph.connections), 0)
        self.assertEqual(report['evidence_records'][0]['state'], 'ambiguous_visible_segment_end_count')

    def test_two_tips_with_branch_still_abstain(self):
        record = deepcopy(self.record); record['branch_cluster_count'] = 1
        self.assertEqual(len(self.adapt(record)[0].connections), 0)

    def test_missing_invalid_and_inconsistent_geometry_fail_closed(self):
        for field in ('candidate_tip_count', 'branch_cluster_count', 'geometry_pair_eligible',
                      'skeleton_component_count', 'candidate_tips_xy', 'endpoint_status'):
            record = deepcopy(self.record); del record[field]
            self.assertEqual(len(self.adapt(record)[0].connections), 0, field)
        record = deepcopy(self.record); record['candidate_tips_xy'] = [[2, 3], [22, 3]]
        self.assertEqual(len(self.adapt(record)[0].connections), 0)
        record = deepcopy(self.record); record['geometry_contract_version'] = 2
        self.assertEqual(len(self.adapt(record)[0].connections), 0)
        for field, value in (('geometry_contract_version', True), ('skeleton_component_count', True),
                             ('branch_cluster_count', False)):
            record = deepcopy(self.record); record[field] = value
            self.assertEqual(len(self.adapt(record)[0].connections), 0)

    def test_report_contract_without_record_metadata_abstains(self):
        record = {'record_id': 'legacy', 'source_score': .9, 'visible_ends_xy': [[3, 3], [21, 3]]}
        self.assertEqual(len(self.adapt(record, geometry_contract_version=1)[0].connections), 0)

    def test_legacy_contract_is_flagged_not_geometry_verified(self):
        record = {'record_id': 'legacy', 'source_score': .9, 'visible_ends_xy': [[3, 3], [21, 3]]}
        graph, report = self.adapt(record)
        self.assertEqual(len(graph.connections), 1)  # preserve old controlled contract
        self.assertFalse(report['evidence_records'][0]['geometry_evidence_verified'])
        self.assertEqual(report['summary']['legacy_geometry_unverified_record_count'], 1)


if __name__ == '__main__': unittest.main()
