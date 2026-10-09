import unittest
from unittest.mock import patch
import run_allport480_teacher_holdouts as runner


class FixedStageGates(unittest.TestCase):
    def check(self,stage,tp,unmatched=0,normal=0,lost=None,lost_original=None):
        return runner.stage_qualifies(stage,dict(trial=dict(tp=tp,unmatched=unmatched)),normal,
            [dict(lost=lost or [],lost_original=lost_original or [])])

    def test_inner_strict_gain_required(self):
        self.assertFalse(self.check('inner',68));self.assertTrue(self.check('inner',69))

    def test_inner_zero_unmatched_and_normal(self):
        self.assertFalse(self.check('inner',79,1));self.assertFalse(self.check('inner',79,normal=1))

    def test_outer_floor_and_unmatched_budget(self):
        self.assertFalse(self.check('outer',39));self.assertTrue(self.check('outer',40,1));self.assertFalse(self.check('outer',55,2))

    def test_no_old_or_original_target_loss(self):
        for stage in ['inner','outer']:
            self.assertFalse(self.check(stage,79,lost=[0]));self.assertFalse(self.check(stage,79,lost_original=[1]))

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):self.check('tuned',100)


class ConditionalEntry(unittest.TestCase):
    def test_failed_or_incomplete_source_never_reads_validation_or_loads_models(self):
        for status in ['rejected','running','failed']:
            source=dict(status=status,qualifies=False)
            with (self.subTest(status=status),patch.object(runner,'load',side_effect=[source,dict(status='pass')]) as load,
                patch.object(runner,'read_current_case',side_effect=AssertionError('forbidden validation image/cache read')),
                patch.object(runner,'read_targets',side_effect=AssertionError('forbidden validation label read'))):
                with self.assertRaisesRegex(ValueError,'strict source pass required'):runner.main()
                self.assertEqual([c.args[0] for c in load.call_args_list],[runner.SOURCE/'report.json',runner.AUDIT])

    def test_independent_rejection_never_reads_validation(self):
        source=dict(status='source_pass_requires_independent_replay_and_fresh_holdouts',qualifies=True)
        audit=dict(status='pass',candidate_source_qualifies=False)
        with patch.object(runner,'load',side_effect=[source,audit]) as load:
            with self.assertRaisesRegex(ValueError,'strict source pass required'):runner.main()
            self.assertEqual(load.call_count,2)


class FixedMembershipTests(unittest.TestCase):
    def setUp(self):
        self.inner=['i%d.JPG'%i for i in range(48)]
        self.outer=['o%d.JPG'%i for i in range(30)]
        self.train=['t%d.JPG'%i for i in range(192)]

    def test_inner_disjoint_exact_accepted_population(self):
        self.assertTrue(runner.check_stage_membership('inner',self.inner,list(reversed(self.inner)),self.train,self.inner))

    def test_inner_source_overlap_rejected(self):
        with self.assertRaises(ValueError):runner.check_stage_membership('inner',self.inner,self.inner,self.train+[self.inner[0]],self.inner)

    def test_same_count_changed_membership_rejected(self):
        changed=self.inner[:-1]+['new.JPG']
        with self.assertRaises(ValueError):runner.check_stage_membership('inner',changed,self.inner,self.train,self.inner)

    def test_changed_frozen_group_rejected(self):
        with self.assertRaises(ValueError):runner.check_stage_membership('inner',self.inner,self.inner,self.train,self.inner[:-1]+['x.JPG'])

    def test_duplicate_rejected(self):
        duplicated=self.outer[:-1]+[self.outer[0]]
        with self.assertRaises(ValueError):runner.check_stage_membership('outer',duplicated,duplicated,self.train,self.inner)

    def test_outer_directory_identity_not_basename_identity(self):
        self.assertTrue(runner.check_stage_membership('outer',self.outer,self.outer,self.train+self.outer,self.inner))

    def test_path_injection_rejected(self):
        names=self.outer[:-1]+['../o29.JPG']
        with self.assertRaises(ValueError):runner.check_stage_membership('outer',names,names,self.train,self.inner)

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):runner.check_stage_membership('all',self.inner,self.inner,self.train,self.inner)


if __name__=='__main__':unittest.main(verbosity=2)
