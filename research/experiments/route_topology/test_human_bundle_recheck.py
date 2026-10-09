import unittest
from copy import deepcopy
from human_bundle_recheck import recheck


class HumanRecheckTests(unittest.TestCase):
    def fixture(self):
        binding={'image_sha256':'a'*64,'image_size':[200,100],'coordinate_frame':'source_image_pixels'}
        scope={'schema_version':1,'kind':'reference_once_visible_lead_attachment','reference_binding':binding,
               'scope_description':'synthetic fixture, not physical evidence','electrical_terminal_pair_established':False,
               'anchors':[{'id':'A','kind':'visible_lead_emergence','bbox_xyxy':[10,10,20,20]},
                          {'id':'B','kind':'wire_entry_socket','bbox_xyxy':[80,10,90,20]}],
               'expected_visible_attachment':['A','B'],'reference_review':{'confirmed':True,'reviewer':'software_fixture', 'evidence_note':'synthetic fixture'}}
        case={'id':'fixture','image_binding':binding,'automatic_comparison':{'decision':'insufficient_evidence'}}
        s={'case_id':'fixture','submission_kind':'software_fixture','image_binding':deepcopy(binding),'reviewer':'software_fixture','evidence_note':'not real human or photo validation',
           'scope_acknowledged':True,'socket_state':'attached','same_bundle_confirmed':True,'route_change':'different',
           'endpoints':[dict(a,identity=a['id'],identity_confirmed=True) for a in scope['anchors']]}
        return scope,case,s

    def test_route_change_does_not_change_attachment(self):
        for route in ['same','different','unknown']:
            scope,c,s=self.fixture();s['route_change']=route
            r=recheck(scope,c,s)
            self.assertEqual(r['human_recheck_decision'],'human_confirmed_same_visible_attachment')
            self.assertEqual(r['automatic_comparison_unchanged']['decision'],'insufficient_evidence')
            self.assertEqual(r['automatic_new_hits'],0)
            self.assertEqual(r['electrical_continuity'],'not_assessed')

    def test_changed_endpoint_and_exposed_socket(self):
        scope,c,s=self.fixture();s['endpoints'][1]['identity']='C'
        self.assertEqual(recheck(scope,c,s)['human_recheck_decision'],'human_confirmed_changed_visible_attachment')
        scope,c,s=self.fixture();s.update(socket_state='contacts_exposed',same_bundle_confirmed=False)
        self.assertEqual(recheck(scope,c,s)['human_recheck_decision'],'human_confirmed_reference_socket_contacts_exposed')

    def test_incomplete_evidence_remains_unknown(self):
        for field,value in [('socket_state','uncertain'),('same_bundle_confirmed',False)]:
            scope,c,s=self.fixture();s[field]=value
            self.assertEqual(recheck(scope,c,s)['human_recheck_decision'],'insufficient_evidence')
        scope,c,s=self.fixture();s['endpoints'][0]['identity_confirmed']=False
        self.assertEqual(recheck(scope,c,s)['human_recheck_decision'],'insufficient_evidence')

    def test_hash_scope_reviewer_and_coordinates_fail_closed(self):
        for mutation in [lambda s:s.update(case_id='other'),lambda s:s['image_binding'].update(image_sha256='b'*64),
                         lambda s:s.update(scope_acknowledged=False),lambda s:s.update(reviewer=''),
                         lambda s:s.update(evidence_note=''),lambda s:s['endpoints'][0].update(bbox_xyxy=[0,0,float('nan'),20]),
                         lambda s:s['endpoints'][0].update(bbox_xyxy=[0,0,201,20]),
                         lambda s:s['endpoints'][0].update(identity_confirmed=1),
                         lambda s:s['endpoints'][0].update(bbox_xyxy=[80,10,90,20])]:
            scope,c,s=self.fixture();mutation(s)
            with self.assertRaises(ValueError):recheck(scope,c,s)

    def test_input_not_mutated_and_reports_independent(self):
        scope,c,s=self.fixture();before=deepcopy((scope,c,s));r=recheck(scope,c,s)
        self.assertEqual((scope,c,s),before)
        r['human_review']['reviewer']='changed'
        self.assertNotEqual(r['human_review']['reviewer'],s['reviewer'])

    def test_invalid_types_and_unlabelled_provenance_rejected(self):
        scope,c,s=self.fixture()
        for invalid in [None,[],True]:
            with self.assertRaises(ValueError):recheck(scope,c,invalid)
        del s['submission_kind']
        with self.assertRaises(ValueError):recheck(scope,c,s)

    def test_identity_outer_whitespace_is_not_attachment_change(self):
        for spaces in [(' A ',' B '), ('\tA\n','\u3000B\u3000')]:
            scope,c,s=self.fixture()
            for endpoint,value in zip(s['endpoints'],spaces):endpoint['identity']=value
            before=deepcopy(s)
            r=recheck(scope,c,s)
            self.assertEqual(r['human_recheck_decision'],'human_confirmed_same_visible_attachment')
            self.assertEqual(r['observed_visible_attachment'],{'visible_lead_emergence':'A','wire_entry_socket':'B'})
            self.assertEqual(s,before)
            self.assertEqual(r['human_review'],before)

    def test_distinct_identity_is_not_fuzzy_matched(self):
        scope,c,s=self.fixture();s['endpoints'][1]['identity']='b'
        self.assertEqual(recheck(scope,c,s)['human_recheck_decision'],'human_confirmed_changed_visible_attachment')

    def test_review_source_controls_display_and_execution_status(self):
        for kind,label,status in [('human_ui_review','人工复核','human_visual_review_recorded'),
                                  ('software_fixture','软件测试（非真实验收）','software_fixture_recorded'),
                                  ('assistant_visual_review','AI 自查（非人类验收）','assistant_visual_review_recorded')]:
            scope,c,s=self.fixture();s['submission_kind']=kind
            r=recheck(scope,c,s)
            self.assertEqual(r['review_source_label'],label)
            self.assertEqual(r['recheck_execution_status'],status)
            self.assertTrue(r['review_result_label'].startswith(label+'：'))
            self.assertEqual(r['automatic_new_hits'],0)


if __name__=='__main__':unittest.main()
