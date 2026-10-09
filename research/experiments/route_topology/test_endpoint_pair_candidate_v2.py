import unittest
import test_visible_bundle_relation as fixture
from endpoint_pair_candidate_v2 import compare_endpoint_pair


class ExpandedEndpointTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.BundleTests()
        self.f.setUp()

    def test_conflicting_exposed_socket_keeps_unknown(self):
        view=self.f.observe(state='socket_contacts_exposed')
        r=compare_endpoint_pair(self.f.scope,self.f.reference,view)
        self.assertEqual(r['decision'],'insufficient_endpoint_evidence')
        self.assertTrue(r['socket_evidence_conflict'])
        self.assertEqual(r['socket_phenotype_observed'],'socket_contacts_exposed')
        self.assertEqual(r['candidate_records'],[])

    def test_native_pair_still_not_physical_identity(self):
        r=compare_endpoint_pair(self.f.scope,self.f.reference,self.f.reference)
        self.assertEqual(r['decision'],'reference_endpoint_pair_candidate')
        self.assertFalse(r['same_physical_wire_confirmed'])
        self.assertFalse(r['socket_evidence_conflict'])


if __name__=='__main__':unittest.main()
