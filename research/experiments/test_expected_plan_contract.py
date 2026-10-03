from copy import deepcopy
import json
from pathlib import Path
import unittest
from expected_plan_contract import make_draft,revision_hash,validate_draft
ROOT=Path(__file__).resolve().parents[1]
class ExpectedDraftTests(unittest.TestCase):
    def setUp(self):
        self.bundle=json.loads((ROOT/'artifacts/local_review_ui_v3_20261002/bundle.json').read_text(encoding='utf-8'))
        self.doc=make_draft(self.bundle,'TEST ONLY','automation_fixture')
    def check(self):self.doc['revision_id']=revision_hash(self.doc);return validate_draft(self.bundle,self.doc)
    def test_unknown_and_empty_different_but_unconfirmed(self):
        unknown=self.doc['revision_id'];self.doc['content']['cases'][0]['expected_connections']=[]
        result=self.check();self.assertNotEqual(unknown,result['revision_id']);self.assertFalse(result['comparison_allowed']);self.assertEqual(result['connection_edges'],[])
    def test_valid_edge_draft_and_child_revision(self):
        row=self.doc['content']['cases'][0];row['selected_port_ids']=['source_entry_1','source_entry_2'];row['port_labels']={p:dict(device_id='TEST',terminal_label='草稿'+p) for p in row['selected_port_ids']};row['expected_connections']=[row['selected_port_ids'][:]]
        previous=self.check()['revision_id'];self.doc['parent_revision_id']=previous;self.doc['author']='中文测试';result=self.check()
        self.assertNotEqual(previous,result['revision_id']);self.assertEqual(result['parent_revision_id'],previous)
    def test_changed_content_without_hash_rejected(self):
        self.doc['author']='changed'
        with self.assertRaises(ValueError):validate_draft(self.bundle,self.doc)
    def test_source_confirmation_and_extra_claims_rejected(self):
        for mutate in (lambda d:d.update(bundle_id='stale'),lambda d:d.update(confirmed=True),lambda d:d.update(comparison_allowed=True),lambda d:d.update(connection_edges=[['a','b']])):
            self.setUp();mutate(self.doc)
            with self.assertRaises(ValueError):self.check()
    def test_bad_scope_self_duplicate_outside_edges(self):
        for edges in ([['source_entry_1','source_entry_1']],[['source_entry_1','source_entry_3']],[['source_entry_1','source_entry_2'],['source_entry_2','source_entry_1']]):
            self.setUp();row=self.doc['content']['cases'][0];row['selected_port_ids']=['source_entry_1','source_entry_2'];row['port_labels']={p:dict(device_id='',terminal_label='') for p in row['selected_port_ids']};row['expected_connections']=edges
            with self.assertRaises(ValueError):self.check()
    def test_missing_case_and_bad_labels(self):
        self.doc['content']['cases'].pop()
        with self.assertRaises(ValueError):self.check()
        self.setUp();self.doc['content']['cases'][0]['port_labels']=[]
        with self.assertRaises(ValueError):self.check()
    def test_blank_author_invalid_parent(self):
        self.doc['author']=' '
        with self.assertRaises(ValueError):self.check()
        self.setUp();self.doc['parent_revision_id']='bad'
        with self.assertRaises(ValueError):self.check()
if __name__=='__main__':unittest.main()
