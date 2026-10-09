from copy import deepcopy
import unittest

import numpy as np

from core import extract_mask
from test_core import mask
from visible_lead_scope import validate_scope, nominate_view


class VisibleLeadScope(unittest.TestCase):
    def setUp(self):
        self.binding={'image_sha256':'a'*64,'image_size':[120,120],'coordinate_frame':'source_image_pixels'}
        self.scope={'schema_version':1,'kind':'reference_once_visible_lead_attachment',
            'reference_binding':self.binding,'scope_description':'Constructed test, not real electrical terminals',
            'anchors':[{'id':'A','kind':'visible_lead_emergence','bbox_xyxy':[16,26,24,34]},
                       {'id':'B','kind':'wire_entry_socket','bbox_xyxy':[96,26,104,34]}],
            'expected_visible_attachment':['A','B'],'electrical_terminal_pair_established':False,
            'reference_review':{'confirmed':False,'reviewer':None,'evidence_note':None}}
        self.record=extract_mask(mask([(20,30),(100,30)]),.9,'r')

    def nominate(self, records=None, **kwargs):
        return nominate_view(self.scope,self.binding,[self.record] if records is None else records,**kwargs)

    def test_visible_attachment_is_never_electrical_success(self):
        result=self.nominate()
        self.assertEqual(len(result['relation_nominations']),1)
        self.assertEqual(result['decision'],'insufficient_evidence')
        self.assertFalse(result['electrical_terminal_pair_established'])
        self.assertEqual(result['physical_new_connections'],0)

    def test_empty_inventory_cannot_mean_normal(self):
        self.assertEqual(self.nominate([])['decision'],'insufficient_evidence')

    def test_branch_not_pruned(self):
        record=extract_mask(mask([(20,30),(100,30),(60,30),(60,80)]),.9,'branch')
        self.assertEqual(self.nominate([record])['relation_nominations'],[])

    def test_boundary_not_cleared(self):
        record=deepcopy(self.record);record['boundary_truncated']=True
        self.assertEqual(self.nominate([record])['relation_nominations'],[])

    def test_low_score_not_silently_accepted(self):
        record=deepcopy(self.record);record['score']=.74
        self.assertEqual(self.nominate([record])['relation_nominations'],[])

    def test_unknown_endpoint_not_nearest_anchor(self):
        record=extract_mask(mask([(25,30),(100,30)]),.9,'outside')
        self.assertEqual(self.nominate([record])['relation_nominations'],[])

    def test_real_reviewer_required_but_not_sufficient(self):
        self.scope['reference_review']['confirmed']=True
        with self.assertRaises(ValueError):self.nominate()
        self.scope['reference_review'].update(reviewer='software_fixture',evidence_note='synthetic only')
        self.assertEqual(self.nominate()['decision'],'insufficient_evidence')

    def test_exact_crop_offset_and_forward_homography(self):
        record=extract_mask(mask([(10,20),(90,20)]),.9,'translated')
        result=self.nominate([record],translation=(15,12),
            inspection_to_reference=np.array([[1,0,-5],[0,1,-2],[0,0,1]]))
        self.assertEqual(len(result['relation_nominations']),1)

    def test_false_tip_or_duplicate_id_invalid(self):
        with self.assertRaises(ValueError):self.nominate([self.record,self.record])
        record=deepcopy(self.record);record['geometry']['tips_xy'][0]=[21,30]
        with self.assertRaises(ValueError):self.nominate([record])

    def test_type_and_nonoverlap_contract(self):
        for mutate in [lambda s:s['anchors'][0].update(kind='electrical_terminal'),
                       lambda s:s['anchors'][1].update(bbox_xyxy=[18,28,26,36]),
                       lambda s:s.update(electrical_terminal_pair_established=True)]:
            scope=deepcopy(self.scope);mutate(scope)
            with self.assertRaises(ValueError):validate_scope(scope,self.binding)

    def test_input_not_mutated(self):
        scope,record=deepcopy(self.scope),deepcopy(self.record)
        self.nominate()
        self.assertEqual(scope,self.scope);self.assertEqual(record,self.record)


if __name__=='__main__':unittest.main()
