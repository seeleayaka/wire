import unittest
from bundle_recheck_plan import plan


class RecheckPlanTests(unittest.TestCase):
    def comparison(self,decision):
        return {'decision':decision,'observation_granularity':'multiwire_bundle_only',
            'physical_new_connections':0,'electrical_correctness':'not_assessed'}

    def test_all_three_actions_are_non_actuating(self):
        expected={'same_visible_bundle_attachment_supported':'record_supported_visible_bundle_scope',
            'visible_socket_attachment_change_supported':'request_human_socket_reinspection',
            'insufficient_evidence':'request_additional_visible_evidence'}
        for decision,kind in expected.items():
            result=plan(self.comparison(decision))
            self.assertEqual(result['action']['kind'],kind);self.assertEqual(result['recheck_execution_status'],'not_performed')
            self.assertFalse(result['physical_actuation_requested']);self.assertFalse(result['automatic_repair_performed'])

    def test_unsupported_claim_or_granularity_rejected(self):
        for change in [{'decision':'electrically_correct'},{'physical_new_connections':1},
                       {'observation_granularity':'single_wire'},{'electrical_correctness':'correct'}]:
            row=self.comparison('insufficient_evidence');row.update(change)
            with self.assertRaises(ValueError):plan(row)


if __name__=='__main__':unittest.main()
