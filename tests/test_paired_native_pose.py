import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from inspection_agent import paired_native_pose


class PortableBackend(unittest.TestCase):
    def setUp(self):
        self.module=paired_native_pose
    def original(self):return dict(status='applied',rescue_hints=[dict(id='protected')],supplementary_hints=[],parents=[1],existing_hints=[2])
    def test_default_off_does_not_touch_new_head(self):
        original=self.original();before=copy.deepcopy(original)
        with patch.object(self.module,'run_paired_median_review',return_value=copy.deepcopy(original)) as parent,patch.object(self.module,'append_native_pose_review') as added:
            result=self.module.run_native_pose_review({},project='synthetic',median_enabled=True,paired_enabled=True)
        self.assertFalse(result['native_pose_policy']['enabled']);added.assert_not_called();self.assertEqual(original,before)
        self.assertTrue(parent.call_args.kwargs['median_enabled']);self.assertTrue(parent.call_args.kwargs['paired_enabled'])
    def test_staging_manifest_cannot_enable_before_acceptance(self):
        with self.assertRaises(ValueError):self.module.validate_release({},dict(manifest='unaccepted'))
    def test_contract_rejects_threshold_and_acceptance_drift(self):
        m=self.module;frozen=dict(manifest='bound',head=m.HEAD_SHA,encoder=m.ENCODER_SHA,accepted={'parent':'protected'},geometry='geometry')
        manifest=dict(policy_id=m.POLICY_ID,head_sha256=m.HEAD_SHA,encoder_sha256=m.ENCODER_SHA,geometry_sha256='geometry',
            feature_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],contexts=[1.5,3.0],probability_gate=.98,
            minimum_unique_checkpoint_votes=3,maximum_primary=5,maximum_extra=5,default_off=True,manual_review_only=True,
            automatic_fault_verdict=False,accepted_runtime=frozen['accepted'],source_gates_passed=True,live_diagnostic_passed=True)
        with patch.object(m,'MANIFEST_SHA','bound'):
            m.validate_release(manifest,frozen)
            for key,value in (('probability_gate',.97),('minimum_unique_checkpoint_votes',2),('live_diagnostic_passed',False),('automatic_fault_verdict',True)):
                changed=dict(manifest);changed[key]=value
                with self.assertRaises(ValueError):m.validate_release(changed,frozen)
    def test_missing_fingerprint_retains_all_existing_fields(self):
        original=self.original();before=copy.deepcopy(original)
        with patch.object(self.module,'native_pose_runtime_fingerprint',side_effect=OSError('synthetic missing snapshot')):
            result=self.module.append_native_pose_review({},original,project='synthetic')
        for key,value in original.items():self.assertEqual(result[key],value)
        self.assertEqual(original,before);self.assertIn('OSError',result['native_pose_policy']['fallback_reason'])
    def test_full_existing_extra_budget_skips_new_inference(self):
        original=self.original();original['supplementary_hints']=[{'id':i} for i in range(5)]
        with patch.object(self.module,'native_pose_runtime_fingerprint') as fingerprint:
            result=self.module.append_native_pose_review({},original,project='synthetic')
        fingerprint.assert_not_called();self.assertTrue(result['native_pose_policy']['shared_budget_full'])
    def test_late_failure_discards_new_hints_and_retains_old_only(self):
        original=self.original();original['supplementary_hints']=[{'id':'old'}];before=copy.deepcopy(original)
        policy=dict(enabled=True,added_hints=2,automatic_fault_verdict=False)
        result=self.module.failed_review(original,policy,ValueError('synthetic late identity drift'))
        self.assertEqual(result['supplementary_hints'],[{'id':'old'}]);self.assertEqual(result['native_pose_policy']['added_hints'],0)
        self.assertTrue(result['native_pose_policy']['discarded_new_hints_on_failure']);self.assertEqual(original,before)


if __name__=='__main__':unittest.main()
