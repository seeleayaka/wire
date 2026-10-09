from copy import deepcopy
import unittest
import numpy as np
from core import extract_mask
from test_core import mask
from visible_lead_scope import nominate_view
from visible_bundle_relation import observe_bundle,compare_bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.binding={'image_sha256':'a'*64,'image_size':[120,120],'coordinate_frame':'source_image_pixels'}
        self.scope={'schema_version':1,'kind':'reference_once_visible_lead_attachment','reference_binding':self.binding,
            'scope_description':'constructed bundle, not physical electrical connections',
            'anchors':[{'id':'A','kind':'visible_lead_emergence','bbox_xyxy':[16,26,24,34]},
                {'id':'B','kind':'wire_entry_socket','bbox_xyxy':[96,26,104,34]}],
            'expected_visible_attachment':['A','B'],'electrical_terminal_pair_established':False,
            'reference_review':{'confirmed':True,'reviewer':'constructed fixture','evidence_note':'not real user confirmation'}}
        self.poses=[{'id':k,'localization_proposal_supported':True,'gates':{'fixture':True},
            'inspection_to_reference_local':np.eye(3).tolist()} for k in ['A','B']]
        self.raw=mask([(20,30),(100,30)])
        self.reference=self.observe(self.raw)

    def observe(self,raw=None,state='mating_body_visible',poses=None,score=.9,verified=True,masks=None):
        return observe_bundle(self.scope,self.binding,self.poses if poses is None else poses,
            [{'raw':self.raw if raw is None else raw,'score':score,'record_id':'r','recipe':'cable_plus_reference_anatomy_box'}]
            if masks is None else masks,state,sam_inventory_verified=verified)

    def compare(self,observed):return compare_bundle(self.scope,self.reference,observed)

    def test_route_shape_not_used(self):
        changed=mask([(20,30),(20,70),(100,70),(100,30)])
        result=self.compare(self.observe(changed))
        self.assertEqual(result['decision'],'same_visible_bundle_attachment_supported')
        self.assertFalse(result['single_wire_connection_confirmed']);self.assertEqual(result['physical_new_connections'],0)

    def test_bundle_branches_do_not_change_single_wire_gate(self):
        branch=mask([(20,30),(100,30),(60,30),(60,80)])
        self.assertEqual(self.compare(self.observe(branch))['decision'],'same_visible_bundle_attachment_supported')
        self.assertEqual(nominate_view(self.scope,self.binding,[extract_mask(branch,.9,'branch')])['relation_nominations'],[])

    def test_broken_native_component_never_bridged(self):
        raw=self.raw.copy();raw[:,50:70]=0
        self.assertEqual(self.compare(self.observe(raw))['decision'],'insufficient_evidence')

    def test_visible_exposure_not_electrical_disconnection(self):
        p=deepcopy(self.poses);p[0]['localization_proposal_supported']=False
        result=self.compare(self.observe(np.zeros((120,120),np.uint8),state='socket_contacts_exposed',poses=p))
        self.assertEqual(result['decision'],'visible_socket_attachment_change_supported')
        self.assertFalse(result['fan_end_relationship_assessed']);self.assertEqual(result['electrical_disconnections_confirmed'],0)

    def test_exposed_conflict_abstains(self):
        self.assertEqual(self.compare(self.observe(state='socket_contacts_exposed'))['decision'],'insufficient_evidence')

    def test_uncertain_and_unverified_abstain(self):
        for r in [self.observe(state='uncertain'),self.observe(verified=False),self.observe(score=.74)]:
            self.assertEqual(self.compare(r)['decision'],'insufficient_evidence')

    def test_bad_local_anchor_gate_not_overridden(self):
        p=deepcopy(self.poses);p[0]['gates']['fixture']=False
        self.assertEqual(self.compare(self.observe(poses=p))['decision'],'insufficient_evidence')

    def test_ambiguous_masks_not_independent_votes(self):
        masks=[{'raw':self.raw,'score':.9,'record_id':k,'recipe':'cable_plus_reference_anatomy_box'} for k in ['r1','r2']]
        result=self.compare(self.observe(masks=masks))
        self.assertEqual(result['decision'],'insufficient_evidence');self.assertEqual(result['model_observer_count'],1)

    def test_empty_inventory_does_not_mean_correct(self):
        self.assertEqual(self.compare(self.observe(masks=[]))['decision'],'insufficient_evidence')

    def test_reference_review_and_binding_required(self):
        scope=deepcopy(self.scope);scope['reference_review'].update(confirmed=False,reviewer=None,evidence_note=None)
        self.assertEqual(compare_bundle(scope,self.reference,self.reference)['decision'],'insufficient_evidence')
        ref=deepcopy(self.reference);ref['source_binding']['image_sha256']='b'*64
        with self.assertRaises(ValueError):compare_bundle(self.scope,ref,self.reference)

    def test_pixel_frame_malformed_pose_and_duplicates_rejected(self):
        poses=deepcopy(self.poses);poses[0]['inspection_to_reference_local']=np.zeros((3,3)).tolist()
        with self.assertRaises(ValueError):self.observe(poses=poses)
        with self.assertRaises(ValueError):observe_bundle(self.scope,self.binding,self.poses,[], 'uncertain',translation=(-1,0))
        masks=[{'raw':self.raw,'score':.9,'record_id':'same','recipe':'cable_plus_reference_anatomy_box'}]*2
        with self.assertRaises(ValueError):self.observe(masks=masks)

    def test_source_pixels_not_mutated(self):
        raw=self.raw.copy();before=raw.copy();self.observe(raw)
        np.testing.assert_array_equal(before,raw)


if __name__=='__main__':unittest.main()
