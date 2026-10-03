from copy import deepcopy
import json
from pathlib import Path
import unittest
from review_readiness import summarize_review

ROOT=Path(__file__).resolve().parents[1]
class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.bundle=json.loads((ROOT/'artifacts/local_review_ui_v2_20261001/bundle.json').read_text(encoding='utf-8'))
        self.review=json.loads((ROOT/'output/playwright/local_review_v2_20261001/automation_fixture_v2.json').read_text(encoding='utf-8'))
        self.maps={c:json.loads((ROOT/f'artifacts/source_entry_drafts_20261001/{c}_entry_draft.json').read_text(encoding='utf-8')) for c in self.bundle['source_versions']}
    def run_summary(self): return summarize_review(self.bundle,self.review,self.maps)
    def test_fixture_has_no_graph(self):
        result=self.run_summary();self.assertEqual(result['connection_edges'],[])
        self.assertEqual(len(result['entries']),6)
        for gate in result['topology_readiness']:
            self.assertIn('test_record_not_operator_evidence',gate['blockers'])
            self.assertFalse(gate['comparison_performed'])
    def test_operator_support_not_connection(self):
        self.review['review_mode']='operator_review'
        result=self.run_summary()
        self.assertIn('local_support_not_connection',result['entries'][0]['reasons'])
        self.assertEqual(result['connection_edges'],[])
    def test_shared_support_queue(self):
        self.review['items'][1]['candidate_reviews'][0].update(state='supported',evidence_note='synthetic test')
        result=self.run_summary()
        self.assertIn('conflicting_support',result['entries'][0]['reasons'])
        self.assertIn('conflicting_support',result['entries'][1]['reasons'])
    def test_reject_all_not_missing(self):
        for c in self.review['items'][3]['candidate_reviews']: c.update(state='rejected',evidence_note='synthetic test')
        result=self.run_summary()
        self.assertIn('all_candidates_rejected_not_missing_wire',result['entries'][3]['reasons'])
        self.assertIn('no_visible_candidate',result['entries'][5]['reasons'])
        self.assertFalse(result['automatic_fault_verdict'])
    def test_confirmed_expected_does_not_invent_observation(self):
        self.review['review_mode']='operator_review'
        for mapping in self.maps.values():
            for port in mapping['ports']: port['confirmed']=True
            mapping['scope']['expected_complete']=True
            mapping['expected_connections']=[]
            mapping['expected_review']=dict(confirmed=True,reviewer='TEST',evidence_note='synthetic test')
        for gate in self.run_summary()['topology_readiness']:
            self.assertEqual(gate['blockers'],['no_verified_complete_cable_observations'])
            self.assertEqual(gate['decision'],'insufficient_evidence')
    def test_stale_source_and_invalid_reviews_rejected(self):
        self.maps['cabinet_1']['image_binding']['image_sha256']='stale'
        with self.assertRaises(ValueError): self.run_summary()
        self.setUp();self.review['connection_edges']=[['x','y']]
        with self.assertRaises(ValueError): self.run_summary()
if __name__=='__main__': unittest.main()
