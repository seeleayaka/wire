import unittest

from deliver_confirmed_bundle_demo import verified_rows


class DeliveryTests(unittest.TestCase):
    def fixture(self):
        decisions = ['same_visible_bundle_attachment_supported',
                     'visible_socket_attachment_change_supported', 'insufficient_evidence']
        report = {'status': 'complete', 'reference_review_confirmed': True,
                  'deployed_to_E_mainline': False, 'new_confirmed_electrical_connections': 0,
                  'electrical_disconnections_confirmed': 0,
                  'cases': [{'id': str(i), 'comparison': {'decision': decision,
                             'observation_granularity': 'multiwire_bundle_only',
                             'physical_new_connections': 0, 'electrical_correctness': 'not_assessed'}}
                            for i, decision in enumerate(decisions)]}
        audit = {'status': 'PASS', 'native_component_anchor_support_independently_replayed': True,
                 'typed_decisions_independently_replayed': True}
        return report, audit

    def test_three_recheck_plans_do_not_claim_execution(self):
        rows = verified_rows(*self.fixture())
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['recheck_plan']['recheck_execution_status'] == 'not_performed' for row in rows))

    def test_unreviewed_running_or_electrical_claim_rejected(self):
        for changes in [{'status': 'running'}, {'reference_review_confirmed': False},
                        {'new_confirmed_electrical_connections': 1}, {'electrical_disconnections_confirmed': 1},
                        {'deployed_to_E_mainline': True}]:
            report, audit = self.fixture()
            report.update(changes)
            with self.assertRaises(ValueError):
                verified_rows(report, audit)

    def test_missing_replay_or_duplicate_cases_rejected(self):
        report, audit = self.fixture()
        audit['typed_decisions_independently_replayed'] = False
        with self.assertRaises(ValueError):
            verified_rows(report, audit)
        report, audit = self.fixture()
        report['cases'][1]['id'] = report['cases'][0]['id']
        with self.assertRaises(ValueError):
            verified_rows(report, audit)


if __name__ == '__main__':
    unittest.main()
