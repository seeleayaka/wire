"""Launch-boundary and vote replay fixtures. No detector calls or dataset reads."""
import unittest
from unittest.mock import patch
from pathlib import Path
import run_allport_common640_source as runner
from audit_allport_common640_source import audit_new_proposal_votes
import audit_allport_common640_source as auditor


class LaunchBoundaryTests(unittest.TestCase):
    def test_existing_attempt_never_resumed_or_overwritten(self):
        with patch.object(Path, 'exists', return_value=True), patch.object(runner, 'load') as load:
            with self.assertRaises(FileExistsError): runner.main()
            load.assert_not_called()

    def test_missing_frozen_auditor_blocks_before_any_input(self):
        with patch.object(Path, 'exists', return_value=False), patch.object(Path, 'is_file', return_value=False), patch.object(runner, 'load') as load:
            with self.assertRaises(ValueError): runner.main()
            load.assert_not_called()

    def test_active_validation_blocks_before_sha_models_or_output(self):
        source = dict(qualifies=True, normal_cues=0, summary={'trial': {'tp': 300, 'unmatched': 4}})
        audit = dict(status='pass', candidate_source_qualifies=True)
        with patch.object(Path, 'exists', return_value=False), patch.object(Path, 'is_file', return_value=True), \
             patch.object(runner, 'load', side_effect=[source, audit, {'status': 'running'}, {'status': 'pending'}]), \
             patch.object(runner, 'sha') as digest, patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(ValueError): runner.main()
            digest.assert_not_called(); mkdir.assert_not_called()

    def test_successful_validation_never_triggers_fallback(self):
        source = dict(qualifies=True, normal_cues=0, summary={'trial': {'tp': 300, 'unmatched': 4}})
        audit = dict(status='pass', candidate_source_qualifies=True)
        dev = dict(status='development_pass_requires_independent_actual_reference_ROI_Qt_SAM', failed_stage=None)
        with patch.object(Path, 'exists', return_value=False), patch.object(Path, 'is_file', return_value=True), \
             patch.object(runner, 'load', side_effect=[source, audit, dev, {'status': 'pass', 'candidate_development_qualifies': True}]), \
             patch.object(runner, 'sha') as digest:
            with self.assertRaises(ValueError): runner.main()
            digest.assert_not_called()


class ProposalReplayTests(unittest.TestCase):
    def setUp(self):
        self.roles = dict(teacher='a'*64, student='b'*64, feature='c'*64)
        self.row = dict(class_id=0, confidence=.9, box_xyxy=[100, 100, 140, 140])
        self.case = dict(new_voter_views=[dict(weight_sha256=d, predictions={'merged_predictions': [self.row]})
                                        for d in self.roles.values()],
                         proposals=[dict(class_id=0, box_xyxy=self.row['box_xyxy'],
                                         semantic_model_vote_sha256=sorted(self.roles.values()),
                                         localization_voter_best_IoU={d: 1. for d in self.roles.values()})])

    def test_three_actual_weights_pass_and_same_weight_extra_view_is_not_vote(self):
        audit_new_proposal_votes(self.case, self.roles)
        self.case['new_voter_views'].append(self.case['new_voter_views'][0])
        audit_new_proposal_votes(self.case, self.roles)

    def test_no_feature_support_rejected_despite_many_teacher_views(self):
        self.case['new_voter_views'] = self.case['new_voter_views'][:2] + self.case['new_voter_views'][:1]*10
        with self.assertRaises(ValueError): audit_new_proposal_votes(self.case, self.roles)

    def test_forged_localization_rank_or_class_rejected(self):
        self.case['proposals'][0]['localization_voter_best_IoU'][self.roles['teacher']] = .99
        with self.assertRaises(ValueError): audit_new_proposal_votes(self.case, self.roles)
        self.case['proposals'][0]['localization_voter_best_IoU'][self.roles['teacher']] = 1.
        self.case['proposals'][0]['class_id'] = 1
        with self.assertRaises(ValueError): audit_new_proposal_votes(self.case, self.roles)


class GTFreeProposalGenerationTests(unittest.TestCase):
    def test_saved_seed_order_and_reference_coverage_must_replay(self):
        case = dict(source_sha256='a'*64, alignment={'source_to_reference_homography': 'fixture'},
                    new_voter_views=[], current={}, proposals=[{'box_xyxy': [100, 100, 120, 120]}])
        generated = [{'box_xyxy': [100, 100, 120, 120]}, {'box_xyxy': [300, 300, 320, 320]}]
        # Explicit software fixture; no real reference/image/model/label reads.
        with patch.object(auditor, 'expected_in_source', return_value=(None, 'valid_fixture')), \
             patch.object(auditor, 'proposals', return_value=generated), \
             patch.object(auditor, 'valid_boxes', return_value=[0]), \
             patch.object(auditor, 'attach', side_effect=lambda rows, *args: rows):
            auditor.audit_proposal_generation(case, None)
            case['proposals'] = generated
            with self.assertRaises(ValueError): auditor.audit_proposal_generation(case, None)


if __name__ == '__main__': unittest.main(verbosity=2)
