import unittest
from copy import deepcopy
from identity_visual_review_contract import validate_draft,compare_review


class ReviewContractTests(unittest.TestCase):
    def fixtures(self):
        draft=dict(reviewer_type='AI',human_confirmed=False,independent_physical_ground_truth=False,
            eligible_for_accuracy_scoring=False,eligible_for_threshold_calibration=False,
            cross_photo_same_physical_wire_identity='unknown',observations=[dict(review_id='P01',socket='contacts_exposed_apparent')])
        manifest=dict(samples=[dict(review_id='P01',case_id='c1',source_sha256='hash')])
        original=dict(cases=[dict(id='c1',source_binding=dict(image_sha256='hash'),decision='insufficient_endpoint_evidence',reason='uncertain')])
        return draft,manifest,original

    def test_visual_difference_is_not_confirmed_false_negative(self):
        draft,manifest,original=self.fixtures();before=deepcopy(original)
        result=compare_review(draft,manifest,original)
        self.assertEqual(result['provisional_socket_difference_ids'],['c1'])
        self.assertFalse(result['cases'][0]['false_negative_confirmed'])
        self.assertEqual(result['accuracy_denominator'],0)
        self.assertIsNone(result['accuracy_rate'])
        self.assertEqual(original,before)

    def test_AI_label_cannot_become_accuracy_or_calibration_GT(self):
        for key in ['independent_physical_ground_truth','eligible_for_accuracy_scoring','eligible_for_threshold_calibration','human_confirmed']:
            draft,manifest,_=self.fixtures();draft[key]=True
            with self.assertRaises(ValueError):validate_draft(draft,manifest)

    def test_cannot_claim_cross_photo_identity(self):
        draft,manifest,_=self.fixtures();draft['cross_photo_same_physical_wire_identity']='same_wire'
        with self.assertRaises(ValueError):validate_draft(draft,manifest)

    def test_missing_or_duplicate_review_ids_rejected(self):
        draft,manifest,_=self.fixtures();draft['observations']*=2
        with self.assertRaises(ValueError):validate_draft(draft,manifest)

    def test_mismatched_image_binding_rejected(self):
        draft,manifest,original=self.fixtures();manifest['samples'][0]['source_sha256']='other'
        with self.assertRaises(ValueError):compare_review(draft,manifest,original)

    def test_existing_exposure_hint_is_retained(self):
        draft,manifest,original=self.fixtures();original['cases'][0]['decision']='reference_socket_exposure_observed'
        self.assertEqual(compare_review(draft,manifest,original)['provisional_socket_difference_ids'],[])


if __name__=='__main__':unittest.main()
