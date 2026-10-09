from copy import deepcopy
import unittest
import numpy as np
from test_core import mask
from visible_bundle_relation import observe_bundle,compare_bundle


class BundleProperties(unittest.TestCase):
    def setUp(self):
        self.binding={'image_sha256':'a'*64,'image_size':[120,120],'coordinate_frame':'source_image_pixels'}
        self.scope={'schema_version':1,'kind':'reference_once_visible_lead_attachment','reference_binding':self.binding,
            'scope_description':'constructed property tests, not physical wiring',
            'anchors':[{'id':'A','kind':'visible_lead_emergence','bbox_xyxy':[16,26,24,34]},
                {'id':'B','kind':'wire_entry_socket','bbox_xyxy':[96,26,104,34]}],
            'expected_visible_attachment':['A','B'],'electrical_terminal_pair_established':False,
            'reference_review':{'confirmed':True,'reviewer':'fixture','evidence_note':'synthetic test only'}}
        self.poses=[{'id':k,'localization_proposal_supported':True,'gates':{'fixture':True},
            'inspection_to_reference_local':np.eye(3).tolist()} for k in ['A','B']]

    def observe(self,raw,poses=None,translation=(0,0)):
        return observe_bundle(self.scope,self.binding,self.poses if poses is None else poses,
            [{'record_id':'r','raw':raw,'score':.9,'recipe':'cable_plus_reference_anatomy_box'}],
            'mating_body_visible',translation=translation,sam_inventory_verified=True)

    def test_exact_original_crop_translation(self):
        raw=np.zeros((80,100),np.uint8);raw[20,10:91]=255
        result=self.observe(raw,translation=(10,10))
        self.assertTrue(result['unique_native_bundle_observation_supported'])

    def test_independent_local_endpoint_planes(self):
        raw=mask([(20,30),(90,30)]);poses=deepcopy(self.poses)
        poses[1]['inspection_to_reference_local']=[[1,0,10],[0,1,0],[0,0,1]]
        self.assertFalse(self.observe(raw)['unique_native_bundle_observation_supported'])
        self.assertTrue(self.observe(raw,poses=poses)['unique_native_bundle_observation_supported'])

    def test_two_native_candidates_are_ambiguous(self):
        raw=np.zeros((120,120),np.uint8);raw[28,20:101]=255;raw[32,20:101]=255
        result=self.observe(raw)
        self.assertEqual(len(result['eligible_native_components']),2)
        self.assertFalse(result['unique_native_bundle_observation_supported'])

    def test_extra_components_are_retained_not_deleted(self):
        raw=mask([(20,30),(100,30)]);raw[80:83,80:83]=255
        result=self.observe(raw);parts=result['mask_audit'][0]['components']
        self.assertEqual(len(parts),2);self.assertEqual(sum(p['pixel_count'] for p in parts),np.count_nonzero(raw))
        self.assertTrue(result['all_original_masks_and_components_retained'])

    def test_native_mask_boundary_is_not_cleared(self):
        raw=mask([(20,30),(100,30)]);raw[0,90]=255
        self.assertFalse(self.observe(raw)['unique_native_bundle_observation_supported'])

    def test_reordered_scope_is_same_relation(self):
        raw=mask([(20,30),(100,30)]);result=self.observe(raw)
        scope=deepcopy(self.scope);scope['anchors'].reverse()
        other=observe_bundle(scope,self.binding,list(reversed(self.poses)),
            [{'record_id':'r','raw':raw,'score':.9,'recipe':'cable_plus_reference_anatomy_box'}],
            'mating_body_visible',sam_inventory_verified=True)
        self.assertEqual(compare_bundle(scope,result,other)['decision'],'same_visible_bundle_attachment_supported')


if __name__=='__main__':unittest.main()
