from copy import deepcopy
import unittest
from local_review_contract import validate_local_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.bundle=dict(bundle_id='fixture',source_versions={'v':'1'},items=[dict(item_id='a',case='c',bbox_xyxy=[1,2,3,4],candidate_record_ids=['r']),dict(item_id='b',case='c',bbox_xyxy=[5,2,7,4],candidate_record_ids=['r'])])
        self.review=dict(schema_version=1,bundle_id='fixture',source_versions={'v':'1'},review_mode='automation_fixture',reviewer='UI_TEST_ONLY',claim_scope='local_visible_relation_only',connection_edges=[],items=[dict(item_id='a',bbox_xyxy=[1,2,3,4],state='supported',record_id='r',evidence_note='fixture only'),dict(item_id='b',bbox_xyxy=[5,2,7,4],state='pending',record_id=None,evidence_note='')])

    def test_valid_never_emits_topology(self):
        result=validate_local_review(self.bundle,self.review)
        self.assertEqual(result['connection_edges'],[])
        self.assertFalse(result['operator_review_verified'])

    def test_shared_support_not_silently_resolved(self):
        self.review['items'][1].update(state='supported',record_id='r',evidence_note='fixture only')
        self.assertEqual(validate_local_review(self.bundle,self.review)['shared_support_item_ids'],['a','b'])

    def test_stale_or_forged(self):
        for key,value in [('bundle_id','other'),('source_versions',{'v':'2'}),('reviewer',''),('connection_edges',[['a','b']]),('review_mode','automatic')]:
            bad=deepcopy(self.review);bad[key]=value
            with self.assertRaises(ValueError):validate_local_review(self.bundle,bad)

    def test_invalid_item(self):
        for key,value in [('bbox_xyxy',[0,0,2,2]),('record_id','other'),('evidence_note',''),('state','connected')]:
            bad=deepcopy(self.review);bad['items'][0][key]=value
            with self.assertRaises(ValueError):validate_local_review(self.bundle,bad)

    def test_missing_duplicate_items(self):
        for rows in [self.review['items'][:1],[self.review['items'][0]]*2]:
            bad=deepcopy(self.review);bad['items']=rows
            with self.assertRaises(ValueError):validate_local_review(self.bundle,bad)

    def test_uncertain_without_candidate(self):
        self.review['items'][0].update(state='uncertain',record_id=None,evidence_note='occluded')
        self.assertEqual(validate_local_review(self.bundle,self.review)['summary']['uncertain'],1)

if __name__=='__main__':unittest.main()
