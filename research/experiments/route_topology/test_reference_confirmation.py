from copy import deepcopy
import unittest
from reference_confirmation import confirm_scope


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.binding={'image_sha256':'a'*64,'image_size':[120,120],'coordinate_frame':'source_image_pixels'}
        self.draft={'schema_version':1,'kind':'reference_once_visible_lead_attachment',
            'reference_binding':self.binding,'scope_description':'constructed visible bundle',
            'anchors':[{'id':'A','kind':'visible_lead_emergence','bbox_xyxy':[16,26,24,34]},
                       {'id':'B','kind':'wire_entry_socket','bbox_xyxy':[96,26,104,34]}],
            'expected_visible_attachment':['A','B'],'electrical_terminal_pair_established':False,
            'reference_review':{'confirmed':False,'reviewer':None,'evidence_note':None}}
        self.approval={'source':'human_user_message_in_current_chat','approved':True,'quote':'yes',
            'question':'Confirm constructed reference?', 'reviewer':'fixture only','client_date_hk':'2026-10-06',
            'reference_binding':self.binding,'expected_visible_attachment':['A','B']}

    def test_only_reference_review_changes_and_draft_retained(self):
        old=deepcopy(self.draft);result=confirm_scope(self.draft,self.binding,self.approval)
        self.assertEqual(old,self.draft);self.assertTrue(result['reference_review']['confirmed'])
        self.assertEqual(result['anchors'],old['anchors']);self.assertFalse(result['electrical_terminal_pair_established'])
        self.assertFalse(result['reference_review']['electrical_continuity_confirmed'])

    def test_different_image_or_relation_rejected(self):
        for field,value in [('reference_binding',{**self.binding,'image_sha256':'b'*64}),
                            ('expected_visible_attachment',['A','C'])]:
            approval=deepcopy(self.approval);approval[field]=value
            with self.assertRaises(ValueError):confirm_scope(self.draft,self.binding,approval)

    def test_no_inferred_human_review(self):
        for field,value in [('source','assistant_visual_review'),('approved',1),('quote',''),('reviewer',None)]:
            approval=deepcopy(self.approval);approval[field]=value
            with self.assertRaises(ValueError):confirm_scope(self.draft,self.binding,approval)


if __name__=='__main__':unittest.main()
