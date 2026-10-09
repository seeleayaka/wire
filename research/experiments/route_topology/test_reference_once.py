from copy import deepcopy
import unittest

import numpy as np

from reference_once import draft_template, map_reference_ports, validate_template


class ReferenceOnce(unittest.TestCase):
    def setUp(self):
        self.binding = {'image_sha256': 'a' * 64, 'image_size': [200, 150],
                        'coordinate_frame': 'source_image_pixels'}
        self.template = draft_template('reference.png', self.binding)
        review = {'confirmed': True, 'reviewer': 'software_fixture_not_real_reviewer',
                  'evidence_note': 'constructed test ports only'}
        self.template.update(ports=[{'id': identity, 'roi_kind': 'wire_entry_port',
            'bbox_xyxy': box, **review} for identity, box in
            [('A', [10, 20, 20, 30]), ('B', [100, 20, 110, 30])]],
            expected_connections=[{'from': 'A', 'to': 'B'}], reference_review=review)

    def run_map(self, matrix=None, reliable=True, template=None):
        return map_reference_ports(self.template if template is None else template,
            self.binding, self.binding, np.eye(3) if matrix is None else matrix,
            {'reliable': reliable})

    def test_draft_never_confirms_empty_reference(self):
        draft = draft_template('source.png', self.binding)
        self.assertFalse(validate_template(draft, self.binding))
        draft['reference_review'] = self.template['reference_review']
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            validate_template(draft, self.binding)

    def test_inverse_direction_and_no_confirmation_inheritance(self):
        matrix = np.array([[1, 0, -5], [0, 1, -7], [0, 0, 1]])
        result = self.run_map(matrix)
        self.assertEqual(result['port_proposals'][0]['bbox_xyxy'], [15, 27, 25, 37])
        self.assertFalse(result['port_proposals'][0]['confirmed'])
        self.assertEqual(result['new_confirmed_connections'], 0)
        self.assertEqual(result['topology_decision'], 'insufficient_evidence')

    def test_no_mutation_or_path_based_answer(self):
        before = deepcopy(self.template)
        original = self.run_map()
        self.assertEqual(before, self.template)
        self.template['reference_image_path'] = 'not_a_fault_label.jpg'
        self.assertEqual(original, self.run_map())

    def test_unreliable_registration_does_not_propose_ports(self):
        self.assertEqual(self.run_map(reliable=False)['port_proposals'], [])
        with self.assertRaises(ValueError):
            self.run_map(reliable=1)

    def test_invalid_transform_not_accepted(self):
        for matrix in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2)):
            with self.assertRaises(ValueError):
                self.run_map(matrix)

    def test_horizon_and_outside_roi_are_not_clipped(self):
        inv = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -15]], dtype=float)
        result = self.run_map(np.linalg.inv(inv))
        self.assertEqual(result['rejections'][0]['reason'], 'projective_horizon_in_roi')
        result = self.run_map(np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]]))
        self.assertEqual(result['rejections'][0]['reason'], 'mapped_roi_outside_inspection')
        self.assertEqual(len(result['port_proposals']), 1)

    def test_reference_binding_drift_rejected(self):
        template = deepcopy(self.template)
        template['reference_binding']['image_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.run_map(template=template)

    def test_unknown_or_duplicate_edge_not_calibration(self):
        for edges in ([{'from': 'A', 'to': 'C'}],
                      [{'from': 'A', 'to': 'B'}, {'from': 'B', 'to': 'A'}]):
            template = deepcopy(self.template)
            template['expected_connections'] = edges
            with self.assertRaises(ValueError):
                self.run_map(template=template)

    def test_boolean_or_body_roi_rejected(self):
        for field, value in [('bbox_xyxy', [True, 20, 30, 40]), ('roi_kind', 'device_body')]:
            template = deepcopy(self.template)
            template['ports'][0][field] = value
            with self.assertRaises(ValueError):
                self.run_map(template=template)


if __name__ == '__main__':
    unittest.main()
