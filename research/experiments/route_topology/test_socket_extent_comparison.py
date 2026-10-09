from copy import deepcopy
import unittest
import numpy as np
from socket_native_extent import observe_with_extent, compare_with_extent
import test_visible_bundle_relation as bundle_fixtures


class ExtentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = bundle_fixtures.BundleTests('test_route_shape_not_used')
        self.fixture.setUp()
        self.reference = self.observe(self.fixture.raw)

    def observe(self, raw, state='mating_body_visible', verified=True):
        f = self.fixture
        return observe_with_extent(f.scope, f.binding, f.poses,
            [{'raw': raw, 'score': .9, 'record_id': 'fixture', 'recipe': 'cable_plus_reference_anatomy_box'}],
            state, sam_inventory_verified=verified)

    def test_visible_body_decision_preserved(self):
        result = compare_with_extent(self.fixture.scope, self.reference, self.reference)
        self.assertEqual(result['decision'], 'same_visible_bundle_attachment_supported')
        self.assertEqual(result['physical_new_connections'], 0)

    def test_exposed_port_speck_not_wire_conflict(self):
        raw = np.zeros((120, 120), np.uint8); raw[30, 100] = 1
        view = self.observe(raw, 'socket_contacts_exposed')
        self.assertTrue(view['high_score_mask_touches_socket'])
        self.assertFalse(view['native_outgoing_component_at_socket'])
        result = compare_with_extent(self.fixture.scope, self.reference, view)
        self.assertEqual(result['decision'], 'visible_socket_attachment_change_supported')
        self.assertFalse(result['fan_end_relationship_assessed'])
        self.assertEqual(result['electrical_disconnections_confirmed'], 0)

    def test_genuine_native_outgoing_component_still_conflicts(self):
        view = self.observe(self.fixture.raw, 'socket_contacts_exposed')
        self.assertTrue(view['native_outgoing_component_at_socket'])
        self.assertEqual(compare_with_extent(self.fixture.scope, self.reference, view)['decision'], 'insufficient_evidence')

    def test_unknown_unverified_and_unconfirmed_not_overridden(self):
        raw = np.zeros((120, 120), np.uint8); raw[30, 100] = 1
        for view in [self.observe(raw, 'uncertain'), self.observe(raw, 'socket_contacts_exposed', False)]:
            self.assertEqual(compare_with_extent(self.fixture.scope, self.reference, view)['decision'], 'insufficient_evidence')
        scope = deepcopy(self.fixture.scope)
        scope['reference_review'].update(confirmed=False, reviewer=None, evidence_note=None)
        self.assertEqual(compare_with_extent(scope, self.reference, self.observe(raw, 'socket_contacts_exposed'))['decision'], 'insufficient_evidence')

    def test_broken_bundle_not_repaired_by_socket_extent(self):
        raw = self.fixture.raw.copy(); raw[:, 50:70] = 0
        result = compare_with_extent(self.fixture.scope, self.reference, self.observe(raw))
        self.assertEqual(result['decision'], 'insufficient_evidence')


if __name__ == '__main__':
    unittest.main()
