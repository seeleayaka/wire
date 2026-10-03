from copy import deepcopy
import unittest
from local_review_contract_v2 import migrate_v1,validate_candidate_review
from test_local_review_contract import ReviewTests


class CandidateTests(unittest.TestCase):
    def setUp(self):
        fixture=ReviewTests();fixture.setUp();self.bundle=fixture.bundle;self.old=fixture.review
        self.bundle['items'][0]['candidate_record_ids'].append('s')
        self.review=migrate_v1(self.bundle,self.old)

    def test_migration_preserves_unselected_pending(self):
        self.assertEqual(self.review['items'][0]['candidate_reviews'][1]['state'],'pending')
        self.assertEqual(validate_candidate_review(self.bundle,self.review)['candidate_summary'],{'supported':1,'pending':2})

    def test_independent_reject(self):
        self.review['items'][0]['candidate_reviews'][1].update(state='rejected',evidence_note='different visible conductor')
        self.assertEqual(validate_candidate_review(self.bundle,self.review)['candidate_summary']['rejected'],1)

    def test_missing_or_duplicate_or_unknown_candidate(self):
        for candidates in [self.review['items'][0]['candidate_reviews'][:1],[self.review['items'][0]['candidate_reviews'][0]]*2,[dict(record_id='unknown',state='pending',evidence_note='')]]:
            bad=deepcopy(self.review);bad['items'][0]['candidate_reviews']=candidates
            with self.assertRaises(ValueError):validate_candidate_review(self.bundle,bad)

    def test_multi_support_flagged(self):
        self.review['items'][0]['candidate_reviews'][1].update(state='supported',evidence_note='fixture only')
        self.assertEqual(validate_candidate_review(self.bundle,self.review)['multiple_supported_candidate_item_ids'],['a'])

    def test_shared_support_flagged(self):
        self.review['items'][1]['candidate_reviews'][0].update(state='supported',evidence_note='fixture only')
        self.assertEqual(validate_candidate_review(self.bundle,self.review)['shared_support_item_ids'],['a','b'])

    def test_bad_versions_notes_scope(self):
        for change in ('version','note','edges','roi'):
            bad=deepcopy(self.review)
            if change=='version':bad['source_versions']={}
            elif change=='note':bad['items'][0]['candidate_reviews'][0]['evidence_note']=''
            elif change=='edges':bad['connection_edges']=[['a','b']]
            else:bad['items'][0]['bbox_xyxy']=[0,0,1,1]
            with self.assertRaises(ValueError):validate_candidate_review(self.bundle,bad)

    def test_entry_uncertainty_no_confirmation(self):
        self.review['items'][1]['entry_review'].update(state='uncertain',evidence_note='occluded')
        self.assertEqual(validate_candidate_review(self.bundle,self.review)['connection_edges'],[])

if __name__=='__main__':unittest.main()
