import unittest
from unittest.mock import patch
from pathlib import Path
import train_allport480_backbone_continuation as runner


class ConditionalTrainingTests(unittest.TestCase):
    def test_live_pass_or_failed_incomplete_never_opens_training(self):
        for status in ['running','starting','failed','development_pass_requires_independent_actual_reference_ROI_Qt_SAM']:
            with self.subTest(status=status),self.assertRaises(ValueError):runner.entry_allowed(dict(status=status),dict(status='pass',candidate_development_qualifies=False))

    def test_completed_rejection_and_independent_pass_required(self):
        r=dict(status='rejected',failed_stage='inner')
        self.assertTrue(runner.entry_allowed(r,dict(status='pass',candidate_development_qualifies=False)))
        for a in [dict(status='failed',candidate_development_qualifies=False),dict(status='pass',candidate_development_qualifies=True),dict(status='pass')]:
            with self.assertRaises(ValueError):runner.entry_allowed(r,a)

    def test_early_guard_precedes_data_or_model_reads(self):
        with patch.object(runner,'load',side_effect=[dict(status='running'),dict(status='pass')]) as load:
            with self.assertRaises(ValueError):runner.main('smoke')
            self.assertEqual([c.args[0] for c in load.call_args_list],[runner.DEV/'report.json',runner.DEV_AUDIT])

    def test_frozen_no_selection_validation_and_no_resume(self):
        options=runner.fixed_options('full',Path('data.yaml'),Path('out'))
        self.assertEqual((options['freeze'],options['epochs'],options['seed'],options['lr0']),(0,2,20261005,.0001))
        self.assertFalse(options['val']);self.assertFalse(options['resume']);self.assertFalse(options['exist_ok'])

    def test_source_balanced_train_diagnostics_are_not_heldouts(self):
        rows=[dict(role='original_replay',source_image=f'train{i}.JPG',image=f'images/train/{i}.jpg',label_count=1 if i<2 else 0) for i in range(4)]
        self.assertEqual(len(runner.choose_train_diagnostics(rows)),4)

    def test_insufficient_sources_rejected(self):
        with self.assertRaises(ValueError):runner.choose_train_diagnostics([dict(role='allport480_positive',source_image='one.JPG',image='one.jpg',label_count=1)]*4)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):runner.fixed_options('sweep',Path('data.yaml'),Path('out'))


if __name__=='__main__':unittest.main(verbosity=2)
