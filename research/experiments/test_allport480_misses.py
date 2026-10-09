from copy import deepcopy
import unittest
from diagnose_allport480_misses import miss_reason


class MissFunnelTests(unittest.TestCase):
    def setUp(self):
        self.roles=dict(teacher='a'*64,student='b'*64,feature='c'*64)
        self.target=dict(class_id=0,box=[10,10,20,20])
        self.row=dict(class_id=0,box_xyxy=[10,10,20,20],confidence=.9)
        self.case=dict(eligible=True,skip_reason=None,
            new_voter_views=[dict(weight_sha256=d,predictions=dict(merged_predictions=[deepcopy(self.row)])) for d in self.roles.values()],
            proposals=[deepcopy(self.row)],probabilities=[[.01,.985,.005]])

    def test_localized_high_is_not_claimed_an_achieved_match(self):
        r=miss_reason(self.case,self.target,self.roles)
        self.assertEqual(r['reason'],'localized_high_semantics_but_novelty_rank_or_budget_blocks')
        self.assertEqual(r['localized_high_semantic_poses'],1)

    def test_many_views_still_one_checkpoint(self):
        self.case['new_voter_views']=[self.case['new_voter_views'][0]]*6
        self.case['proposals']=[];self.case['probabilities']=[]
        r=miss_reason(self.case,self.target,self.roles)
        self.assertEqual(r['unique_checkpoint_IoU50_supports'],1)
        self.assertEqual(r['reason'],'fewer_than_three_actual_checkpoint_supports')

    def test_semantics_not_overridden_by_geometry_support(self):
        self.case['probabilities']=[[.1,.89,.01]]
        self.assertEqual(miss_reason(self.case,self.target,self.roles)['reason'],'frozen_paired_semantics_blocks_localized_pose')

    def test_valid_localized_pose_is_not_mislabeled_missing_direct_GT_vote(self):
        self.case['proposals'][0]['box_xyxy']=[8,10,18,20]
        self.case['probabilities']=[[.1,.89,.01]]
        for view in self.case['new_voter_views']:
            view['predictions']['merged_predictions'][0]['box_xyxy']=[6,10,16,20]
        result=miss_reason(self.case,self.target,self.roles)
        self.assertEqual(result['unique_checkpoint_IoU50_supports'],0)
        self.assertEqual(result['localized_valid_poses'],1)
        self.assertEqual(result['reason'],'frozen_paired_semantics_blocks_localized_pose')

    def test_full_budget_or_registration_abstention_first(self):
        for reason in ['shared_budget_full','unreliable_or_unavailable_reference_alignment']:
            self.case['eligible']=False;self.case['skip_reason']=reason
            self.assertEqual(miss_reason(self.case,self.target,self.roles)['reason'],'policy_'+reason)

    def test_missing_localization_pose_and_class_awareness(self):
        self.case['proposals'][0]['class_id']=1
        self.assertEqual(miss_reason(self.case,self.target,self.roles)['reason'],'no_valid_class_IoU50_pose_after_geometry_context')

    def test_inputs_not_mutated(self):
        before=deepcopy(self.case);miss_reason(self.case,self.target,self.roles);self.assertEqual(self.case,before)


if __name__=='__main__':unittest.main(verbosity=2)
