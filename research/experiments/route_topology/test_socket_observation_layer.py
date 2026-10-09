import unittest
from copy import deepcopy
from socket_observation_layer import summarize_socket_observation


class SocketLayerTests(unittest.TestCase):
    def row(self):
        return dict(local_anchor_support={'FAN_CPU':True,'FAN_LEAD':False},
            socket_phenotype_observed='socket_contacts_exposed',socket_evidence_conflict=True,
            native_inventory_verified=True,decision='insufficient_endpoint_evidence',candidate_records=[])

    def test_conflict_does_not_hide_observation_or_confirm_fault(self):
        row=self.row();before=deepcopy(row);output=summarize_socket_observation(row,True)
        self.assertEqual(output['socket_observation']['state'],'socket_contacts_exposed_observed')
        self.assertEqual(output['socket_observation']['disposition'],'manual_review_conflicting_evidence')
        self.assertEqual(output['decision'],'insufficient_endpoint_evidence')
        self.assertFalse(output['socket_observation']['automatic_fault_confirmed'])
        self.assertEqual(row,before)

    def test_missing_SAM_does_not_hide_positive_socket_only_evidence(self):
        row=self.row();row['native_inventory_verified']=False;row['socket_evidence_conflict']=False
        self.assertEqual(summarize_socket_observation(row,True)['socket_observation']['state'],'socket_contacts_exposed_observed')

    def test_unknown_classifier_not_rescued_by_AI_draft(self):
        row=self.row();row['socket_phenotype_observed']='uncertain'
        self.assertEqual(summarize_socket_observation(row,True)['socket_observation']['state'],'unknown')

    def test_bad_source_or_socket_pose_abstains(self):
        self.assertEqual(summarize_socket_observation(self.row(),False)['socket_observation']['state'],'unknown')
        row=self.row();row['local_anchor_support']['FAN_CPU']=False
        self.assertEqual(summarize_socket_observation(row,True)['socket_observation']['state'],'unknown')

    def test_existing_candidate_kept(self):
        row=self.row();row['decision']='reference_endpoint_pair_candidate';row['socket_phenotype_observed']='mating_body_visible'
        row['socket_evidence_conflict']=False;row['candidate_records']=[dict(record_id='r')]
        output=summarize_socket_observation(row,True)
        self.assertEqual(output['candidate_records'],row['candidate_records'])
        self.assertEqual(output['decision'],row['decision'])


if __name__=='__main__':unittest.main()
