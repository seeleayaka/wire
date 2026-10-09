import unittest
import numpy as np
from copy import deepcopy
import test_visible_bundle_relation as fixture
from endpoint_pair_candidate_v3 import compare_endpoint_pair


class SingleSocketPreservationTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.BundleTests()
        self.f.setUp()

    def test_missing_lead_does_not_remove_existing_socket_evidence(self):
        poses=deepcopy(self.f.poses)
        poses[0]['localization_proposal_supported']=False
        view=self.f.observe(np.zeros_like(self.f.raw),state='socket_contacts_exposed',poses=poses)
        result=compare_endpoint_pair(self.f.scope,self.f.reference,view)
        self.assertEqual(result['decision'],'reference_socket_exposure_observed')
        self.assertTrue(result['single_endpoint_observation_only'])
        self.assertEqual(result['candidate_records'],[])
        self.assertFalse(result['same_physical_wire_confirmed'])

    def test_conflict_still_unknown(self):
        view=self.f.observe(state='socket_contacts_exposed')
        result=compare_endpoint_pair(self.f.scope,self.f.reference,view)
        self.assertEqual(result['decision'],'insufficient_endpoint_evidence')
        self.assertTrue(result['socket_evidence_conflict'])


if __name__=='__main__':unittest.main()
