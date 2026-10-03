import copy,unittest
import numpy as np
from visible_evidence_duplicates import audit_duplicates,endpoint_port_diagnostics
def record(name,eligible=True):return dict(record_id=name,geometry_pair_eligible=eligible,visible_ends_xy=[[1,1],[4,1]])
class EvidenceTests(unittest.TestCase):
    def test_exact_evidence_not_connections(self):
        mask=np.ones((5,5),bool);rows=[record('a'),record('b')];before=copy.deepcopy(rows)
        result=audit_duplicates([mask,mask.copy()],rows)
        self.assertEqual(result['exact_duplicate_redundancy'],1)
        self.assertEqual(result['eligible_exact_mask_groups'],1);self.assertEqual(result['automatic_connections'],[])
        self.assertEqual(rows,before)
    def test_near_overlap_not_deduplicated(self):
        a=np.ones((10,10),bool);b=a.copy();b[0,:]=False
        result=audit_duplicates([a,b],[record('a'),record('b')])
        self.assertEqual(result['exact_mask_groups'],2);self.assertEqual(len(result['high_overlap_pairs']),1)
    def test_bbox_similarity_is_not_mask_similarity(self):
        a=np.eye(5,dtype=bool);b=np.fliplr(a)
        self.assertEqual(audit_duplicates([a,b],[record('a'),record('b')])['high_overlap_pairs'],[])
    def test_truncated_evidence_not_eligible(self):
        a=np.ones((5,5),bool)
        result=audit_duplicates([a,a],[record('a',False),record('b')])
        self.assertEqual(result['eligible_records'],1);self.assertEqual(result['eligible_exact_mask_groups'],1)
    def test_bad_shapes_ids_and_empty(self):
        for masks,rows in (([np.ones((2,2)),np.ones((3,3))],[record('a'),record('b')]),
            ([np.zeros((2,2))],[record('a')]),([np.ones((2,2))]*2,[record('a'),record('a')])):
            with self.assertRaises(ValueError):audit_duplicates(masks,rows)
    def test_no_nearest_assignment(self):
        result=endpoint_port_diagnostics([record('a')],[dict(id='p',bbox_xyxy=[7,7,9,9],confirmed=False)])[0]
        self.assertIsNone(result['assignment']);self.assertEqual(result['endpoint_contacts'],[])
    def test_contact_not_confirmation(self):
        result=endpoint_port_diagnostics([record('a')],[dict(id='p',bbox_xyxy=[0,0,2,2],confirmed=False)])[0]
        self.assertEqual(len(result['endpoint_contacts']),1);self.assertIsNone(result['assignment'])
    def test_empty_batch(self):self.assertEqual(audit_duplicates([],[])['exact_mask_groups'],0)
    def test_contained_fragment_not_merged(self):
        a=np.ones((10,10),bool);b=np.zeros((10,10),bool);b[:3,:]=True
        result=audit_duplicates([a,b],[record('a'),record('b')])
        self.assertEqual(len(result['contained_overlap_pairs']),1)
        self.assertEqual(result['exact_mask_groups'],2)
if __name__=='__main__':unittest.main()
