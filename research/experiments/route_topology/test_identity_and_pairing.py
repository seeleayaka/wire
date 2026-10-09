import unittest
from copy import deepcopy

import numpy as np

from core import extract_mask
from identity_ocr import inverse_rotation, summarize_readings
from paired_evidence import project, propose_correspondences
from test_core import mask


class Pairing(unittest.TestCase):
    def setUp(self):
        self.a=extract_mask(mask([(20,30),(100,30)]),.9,'a')
        self.b=extract_mask(mask([(20,30),(20,60),(100,60),(100,30)]),.9,'b')

    def pair(self,a=None,b=None,h=None,reliable=True):
        return propose_correspondences(a or [self.a],b or [self.b], np.eye(3) if h is None else h,
                                        {'reliable':reliable},[120,120])

    def test_same_endpoints_changed_route_proposal_not_electrical_verdict(self):
        r=self.pair()
        self.assertEqual(len(r['proposals']),1)
        self.assertTrue(r['proposals'][0]['path_shape_difference_detected'])
        self.assertFalse(r['proposals'][0]['port_identity_known'])
        self.assertEqual(r['automatic_connections'],[])
        self.assertFalse(r['original_visual_cues_suppressed'])

    def test_reversed_tip_order_matches(self):
        b=deepcopy(self.b)
        b['geometry']['tips_xy']=b['geometry']['tips_xy'][::-1]
        self.assertEqual(len(self.pair(b=[b])['proposals']),1)

    def test_wrong_end_not_matched(self):
        b=extract_mask(mask([(20,30),(100,80)]),.9,'wrong')
        self.assertEqual(self.pair(b=[b])['proposals'],[])

    def test_low_confidence_not_used(self):
        b=deepcopy(self.b);b['score']=.749
        self.assertEqual(self.pair(b=[b])['proposals'],[])

    def test_unreliable_registration_rejects(self):
        self.assertEqual(self.pair(reliable=False)['proposals'],[])

    def test_ambiguous_same_endpoints_different_masks_do_not_pick_one(self):
        c=extract_mask(mask([(20,30),(20,70),(100,70),(100,30)]),.99,'other')
        self.assertEqual(self.pair(b=[self.b,c])['proposals'],[])

    def test_exact_duplicate_not_extra_vote(self):
        c=deepcopy(self.b);c['record_id']='duplicate'
        r=self.pair(b=[self.b,c])
        self.assertEqual(len(r['proposals']),1)
        self.assertEqual(r['proposals'][0]['independent_model_votes'],1)

    def test_valid_registration_transform(self):
        b=extract_mask(mask([(25,35),(25,65),(105,65),(105,35)]),.9,'moved')
        h=np.array([[1,0,-5],[0,1,-5],[0,0,1.]])
        self.assertEqual(len(self.pair(b=[b],h=h)['proposals']),1)

    def test_point_at_infinity_rejected(self):
        with self.assertRaises(ValueError):
            project([[1,1]],[[1,0,0],[0,1,0],[0,0,0]])

    def test_same_image_has_no_route_change(self):
        r=self.pair(b=[self.a])
        self.assertFalse(r['proposals'][0]['path_shape_difference_detected'])

    def test_inputs_not_modified(self):
        a,b=deepcopy(self.a),deepcopy(self.b)
        self.pair()
        self.assertEqual(a,self.a);self.assertEqual(b,self.b)


def reading(identity,text,score=.95,box=None,view=0):
    return {'record_id':identity,'text':text,'score':score,'bbox_xyxy':box or [10,10,30,20],
            'view_quarter_turns':view}


class OCR(unittest.TestCase):
    def test_all_rotations_round_trip_pixel_coordinates(self):
        original=np.array([[0,0],[20,40],[139,99]],float)
        w,h=140,100
        for k in range(4):
            x,y=original[:,0],original[:,1]
            rotated=[original,np.column_stack((y,w-1-x)),
                     np.column_stack((w-1-x,h-1-y)),np.column_stack((h-1-y,x))][k]
            recovered=inverse_rotation(rotated,w,h,k,2)
            np.testing.assert_allclose(recovered,original/2)

    def test_four_views_are_one_observer(self):
        r=summarize_readings([reading(str(k),'X1:3',view=k) for k in range(4)])
        self.assertEqual(len(r['groups']),1)
        self.assertEqual(r['groups'][0]['model_observer_count'],1)
        self.assertEqual(r['groups'][0]['view_count'],4)
        self.assertFalse(r['groups'][0]['confirmed'])

    def test_conflicting_ocr_readings_not_repaired(self):
        r=summarize_readings([reading('a','O1'),reading('b','01',view=1)])
        self.assertFalse(r['groups'][0]['ocr_high_score_consistent'])
        self.assertEqual(r['groups'][0]['text_variants'],['01','O1'])

    def test_repeated_text_not_a_wire_connection(self):
        r=summarize_readings([reading('a','L1'),reading('b','L1',box=[50,50,70,60])])
        self.assertTrue(r['repeated_text_groups'])
        self.assertEqual(r['automatic_connections'],[])
        self.assertEqual(r['confirmed_port_identities'],[])

    def test_low_score_nomination_not_promoted(self):
        self.assertFalse(summarize_readings([reading('a','12',score=.89)])['groups'][0]['ocr_high_score_consistent'])

    def test_malformed_input_rejects(self):
        with self.assertRaises(ValueError):
            inverse_rotation([[np.nan,1]],10,10,1,2)
        with self.assertRaises(ValueError):
            summarize_readings([reading('a','L1',score=float('nan'))])

    def test_empty_has_no_identity(self):
        self.assertEqual(summarize_readings([])['confirmed_port_identities'],[])

    def test_inputs_not_mutated(self):
        a=[reading('a','X1:1')];saved=deepcopy(a)
        summarize_readings(a)
        self.assertEqual(a,saved)


if __name__=='__main__':
    unittest.main(verbosity=2)
