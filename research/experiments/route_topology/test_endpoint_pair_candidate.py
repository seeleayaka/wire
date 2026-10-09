from copy import deepcopy
import unittest
import numpy as np
import test_visible_bundle_relation as bundle_fixture
from endpoint_pair_candidate import compare_endpoint_pair


class EndpointPairTests(unittest.TestCase):
    def setUp(self):
        self.f = bundle_fixture.BundleTests()
        self.f.setUp()

    def compare(self, view):
        return compare_endpoint_pair(self.f.scope,self.f.reference,view)

    def test_occluded_mask_is_only_record_candidate(self):
        raw = self.f.raw.copy()
        raw[:,50:70] = 0
        view = self.f.observe(raw)
        result = self.compare(view)
        self.assertEqual(result['decision'],'reference_endpoint_pair_candidate')
        self.assertEqual(result['middle_connection'],'unobserved')
        self.assertEqual(result['automatic_topology_comparison']['decision'],'insufficient_evidence')
        self.assertFalse(result['same_physical_wire_confirmed'])
        self.assertEqual(len(result['candidate_records']),1)
        self.assertEqual(result['candidate_records'][0]['anchor_components'],{'A':[1],'B':[2]})

    def test_position_shift_local_mapping_not_absolute_coordinate(self):
        raw = np.zeros_like(self.f.raw)
        raw[10:,10:] = self.f.raw[:-10,:-10]
        raw[:,60:80] = 0
        poses = deepcopy(self.f.poses)
        for p in poses: p['inspection_to_reference_local']=[[1,0,-10],[0,1,-10],[0,0,1]]
        result = self.compare(self.f.observe(raw,poses=poses))
        self.assertEqual(result['decision'],'reference_endpoint_pair_candidate')

    def test_two_different_mask_instances_never_paired(self):
        left,right=self.f.raw.copy(),self.f.raw.copy()
        left[:,50:]=0
        right[:,:70]=0
        masks=[dict(raw=m,score=.9,record_id=i,recipe='cable_plus_reference_anatomy_box')
               for m,i in [(left,'left'),(right,'right')]]
        self.assertEqual(self.compare(self.f.observe(masks=masks))['decision'],'insufficient_endpoint_evidence')

    def test_two_qualifying_instances_ambiguous(self):
        masks=[dict(raw=self.f.raw,score=.9,record_id=i,recipe='cable_plus_reference_anatomy_box') for i in ['x','y']]
        self.assertEqual(self.compare(self.f.observe(masks=masks))['decision'],'insufficient_endpoint_evidence')

    def test_exposed_is_not_same_attachment(self):
        result=self.compare(self.f.observe(np.zeros_like(self.f.raw),state='socket_contacts_exposed'))
        self.assertEqual(result['decision'],'reference_socket_exposure_observed')
        self.assertFalse(result['same_physical_wire_confirmed'])

    def test_uncertain_pose_inventory_and_score_reject(self):
        poses=deepcopy(self.f.poses)
        poses[0]['gates']['fixture']=False
        for view in [self.f.observe(poses=poses),self.f.observe(verified=False),
                     self.f.observe(state='uncertain'),self.f.observe(score=.74)]:
            self.assertEqual(self.compare(view)['decision'],'insufficient_endpoint_evidence')

    def test_topology_and_inputs_unchanged_even_when_same_instance_wrong(self):
        view=self.f.observe()
        before=deepcopy(view)
        result=self.compare(view)
        self.assertEqual(view,before)
        self.assertEqual(result['electrical_continuity'],'not_assessed')
        self.assertFalse(result['same_physical_wire_confirmed'])
        self.assertEqual(result['physical_new_connections'],0)

    def test_reference_scope_binding_checked(self):
        reference=deepcopy(self.f.reference)
        reference['source_binding']['image_sha256']='b'*64
        with self.assertRaises(ValueError):compare_endpoint_pair(self.f.scope,reference,self.f.reference)


if __name__=='__main__':unittest.main()
