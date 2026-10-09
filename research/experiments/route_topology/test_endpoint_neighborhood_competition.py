import unittest
from copy import deepcopy
import numpy as np
from endpoint_neighborhood_competition import assess_competition


class CompetitionTests(unittest.TestCase):
    def record(self,id='a',offset=0,score=.9,state='reference_color_supported'):
        mask=np.zeros((30,30),bool);mask[8+offset:12+offset,5:25]=True
        return dict(record_id=id,score=score,endpoints={'A':dict(matched_pixels=mask,appearance_state=state)})

    def test_different_local_footprints_request_review_not_foreign_wire(self):
        result=assess_competition([self.record(),self.record('b',8)],['A'])
        self.assertTrue(result['review_needed'])
        self.assertEqual(result['endpoints']['A']['distinct_footprints'],2)
        self.assertFalse(result['same_physical_wire_confirmed'])
        self.assertTrue(result['different_masks_are_not_proven_different_wires'])

    def test_exact_duplicate_footprints_do_not_count_twice(self):
        result=assess_competition([self.record(),self.record('b')],['A'])
        self.assertFalse(result['review_needed'])
        self.assertEqual(result['endpoints']['A']['footprints'][0]['native_record_ids'],['a','b'])
        self.assertEqual(result['independent_model_observer_count'],1)

    def test_low_score_or_unknown_appearance_not_positive_support(self):
        result=assess_competition([self.record(score=.74),self.record('b',state='insufficient_colored_evidence')],['A'])
        self.assertEqual(result['endpoints']['A']['distinct_footprints'],0)
        self.assertFalse(result['same_physical_wire_confirmed'])

    def test_permutation_invariance(self):
        rows=[self.record(),self.record('b',4),self.record('c')]
        self.assertEqual(assess_competition(rows,['A']),assess_competition(rows[::-1],['A']))

    def test_no_input_mutation(self):
        rows=[self.record()];before=deepcopy(rows)
        assess_competition(rows,['A'])
        np.testing.assert_array_equal(rows[0]['endpoints']['A']['matched_pixels'],before[0]['endpoints']['A']['matched_pixels'])

    def test_coordinate_shape_mismatch_fails_closed(self):
        a=self.record();b=self.record('b');b['endpoints']['A']['matched_pixels']=np.ones((29,30),bool)
        with self.assertRaises(ValueError):assess_competition([a,b],['A'])

    def test_duplicate_record_id_fails_closed(self):
        with self.assertRaises(ValueError):assess_competition([self.record(),self.record(offset=4)],['A'])

    def test_invalid_score_or_unbound_endpoint_rejected(self):
        with self.assertRaises(ValueError):assess_competition([self.record(score=float('nan'))],['A'])
        with self.assertRaises(ValueError):assess_competition([self.record()],['B'])

    def test_empty_supported_mask_is_not_confirmation(self):
        row=self.record();row['endpoints']['A']['matched_pixels'][:]=False
        result=assess_competition([row],['A'])
        self.assertEqual(result['endpoints']['A']['distinct_footprints'],0)
        self.assertFalse(result['same_physical_wire_confirmed'])


if __name__=='__main__':unittest.main()
