from copy import deepcopy
import unittest
import numpy as np
from label_attachment import canonical_masks,polygon_pixels,attach_polygon,nominate,mask_context_crop


def row(text='A12',score=.95,record_id='r1',polygon=None):
    p=np.asarray(polygon if polygon is not None else [[10,10],[20,10],[20,15],[10,15]],float)
    return {'text':text,'score':score,'record_id':record_id,'view_quarter_turns':0,
            'polygon_source_xy':p.tolist(),
            'bbox_xyxy':[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]}


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        self.size=[80,60]
        self.mask=np.zeros((60,80),bool);self.mask[8:20,8:23]=True
        self.masks=[{'id':'a','pixels':self.mask,'topology_geometry_eligible':False}]

    def test_unique_is_unconfirmed(self):
        result=nominate([row()],self.masks,self.size)
        g=result['groups'][0]
        self.assertTrue(g['high_score_consistent_text_on_mask'])
        self.assertEqual(g['nominated_mask_id'],'a')
        self.assertFalse(g['confirmed']);self.assertIsNone(g['wire_identity'])
        self.assertEqual(result['automatic_connections'],[])

    def test_nearby_mask_is_not_nearest_assignment(self):
        other=np.roll(self.mask,30,axis=1)
        result=nominate([row()],[{'id':'far','pixels':other}],self.size)
        self.assertIsNone(result['groups'][0]['nominated_mask_id'])

    def test_any_second_intersection_is_ambiguous(self):
        other=np.zeros_like(self.mask);other[10,10]=True
        result=nominate([row()],self.masks+[{'id':'b','pixels':other}],self.size)
        self.assertEqual(result['reading_memberships']['r1']['state'],'ambiguous_multiple_masks')

    def test_exact_duplicates_are_one_mask_not_two_votes(self):
        result=nominate([row()],self.masks+[{'id':'copy','pixels':self.mask.copy(),'topology_geometry_eligible':True}],self.size)
        self.assertEqual(result['canonical_mask_count'],1)
        self.assertEqual(result['groups'][0]['sam_model_observer_count'],1)
        self.assertFalse(result['reading_memberships']['r1']['supports'][0]['topology_geometry_eligible'])

    def test_overlap_below_half_rejects(self):
        tiny=np.zeros_like(self.mask);tiny[10,10]=True
        result=nominate([row()],[{'id':'tiny','pixels':tiny}],self.size)
        self.assertEqual(result['reading_memberships']['r1']['state'],'insufficient_mask_overlap')

    def test_no_token_repair(self):
        result=nominate([row(text='O0I1')],self.masks,self.size)
        self.assertEqual(result['groups'][0]['text_group']['best_text'],'O0I1')

    def test_conflicting_views_reject_high_nomination(self):
        result=nominate([row(),row('A13',.8,'r2')],self.masks,self.size)
        self.assertFalse(result['groups'][0]['high_score_consistent_text_on_mask'])

    def test_low_score_remains_low(self):
        self.assertFalse(nominate([row(score=.89)],self.masks,self.size)['groups'][0]['high_score_consistent_text_on_mask'])

    def test_same_text_on_distinct_masks_is_not_same_wire(self):
        b=np.roll(self.mask,30,axis=1)
        shifted=(np.array(row()['polygon_source_xy'])+[30,0]).tolist()
        result=nominate([row(),row(record_id='r2',polygon=shifted)],self.masks+[{'id':'b','pixels':b}],self.size)
        self.assertEqual(result['same_text_different_masks'],{'A12':['a','b']})
        self.assertTrue(all(not g['text_identity_usable_automatically'] for g in result['groups']))

    def test_polygon_not_bbox_controls_overlap(self):
        p=[[10,10],[20,20],[15,25],[5,15]]
        outside=np.zeros_like(self.mask);outside[10,5]=True
        self.assertEqual(attach_polygon(p,canonical_masks([{'id':'b','pixels':outside}],self.size),self.size)['state'],'no_mask_overlap')

    def test_bad_polygon_rejected(self):
        for p in [[[10,10]]*4,[[-1,10],[20,10],[20,15],[10,15]],[[10,10],[20,15],[20,10],[10,15]],[[float('nan'),10],[20,10],[20,15],[10,15]]]:
            with self.subTest(p=p),self.assertRaises(ValueError):
                polygon_pixels(p,self.size)

    def test_empty_inputs_and_empty_mask(self):
        self.assertEqual(nominate([],[],self.size)['groups'],[])
        self.assertIsNone(mask_context_crop(np.zeros((60,80)),self.size))

    def test_frame_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            canonical_masks([{'id':'a','pixels':np.zeros((2,2))}],self.size)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            nominate([row(),row()],self.masks,self.size)
        with self.assertRaises(ValueError):
            canonical_masks(self.masks+self.masks,self.size)

    def test_box_polygon_mismatch_rejected(self):
        r=row();r['bbox_xyxy'][0]-=1
        with self.assertRaises(ValueError):
            nominate([r],self.masks,self.size)

    def test_input_immutable(self):
        readings=[row()];before=deepcopy(readings);pixels=self.mask.copy()
        nominate(readings,self.masks,self.size)
        self.assertEqual(readings,before);np.testing.assert_array_equal(self.mask,pixels)

    def test_translation_invariance(self):
        shifted=(np.array(row()['polygon_source_xy'])+[20,20]).tolist()
        shifted_mask=np.roll(np.roll(self.mask,20,0),20,1)
        result=nominate([row(polygon=shifted)],[{'id':'a','pixels':shifted_mask}],self.size)
        self.assertTrue(result['groups'][0]['high_score_consistent_text_on_mask'])

    def test_context_padding_generic_and_frame_bounded(self):
        self.assertEqual(mask_context_crop(self.mask,self.size),[2,2,29,26])
        edge=np.zeros_like(self.mask);edge[:5,:5]=True
        self.assertEqual(mask_context_crop(edge,self.size),[0,0,9,9])


if __name__=='__main__':
    unittest.main()
