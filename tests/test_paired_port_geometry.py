import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inspection_agent import paired_port_geometry as backend
from inspection_agent import paired_port_features as features


def baseline(status='applied'):
    return dict(status=status,parents=[{'bbox_xyxy':[1,2,3,4]}],existing_hints=[],rescue_hints=[],
                supplementary_hints=[{'box':{'old':True}}],decision='possible_difference_manual_review')


class PairedGeometryGuards(unittest.TestCase):
    def test_default_off_no_new_model_reads(self):
        old=baseline();before=copy.deepcopy(old)
        with patch.object(backend,'run_resolution_plug_review',return_value=old),patch.object(
                backend,'paired_runtime_fingerprint',side_effect=AssertionError('no new model')):
            result=backend.run_paired_geometry_review({},project='unused')
        self.assertFalse(result['paired_geometry_policy']['enabled'])
        for key in before:self.assertEqual(result[key],before[key])

    def test_upstream_abstention_and_full_budget_preserved(self):
        controls=[baseline('fallback'),baseline()];controls[1]['supplementary_hints']*=5
        for old in controls:
            before=copy.deepcopy(old)
            with patch.object(backend,'run_resolution_plug_review',return_value=old),patch.object(
                    backend,'paired_runtime_fingerprint',side_effect=AssertionError('no new model')):
                result=backend.run_paired_geometry_review({},project='unused',paired_enabled=True)
            for key in before:self.assertEqual(result[key],before[key])
            self.assertEqual(result['paired_geometry_policy']['added_hints'],0)

    def test_provenance_failure_retains_all_current_hints(self):
        old=baseline();before=copy.deepcopy(old)
        bad=dict(manifest='changed',head=backend.HEAD_SHA,encoder=backend.ENCODER_SHA)
        with patch.object(backend,'run_resolution_plug_review',return_value=old),patch.object(
                backend,'paired_runtime_fingerprint',return_value=bad):
            result=backend.run_paired_geometry_review({},project='unused',paired_enabled=True)
        self.assertIn('paired_model_provenance_mismatch',result['paired_geometry_policy']['fallback_reason'])
        for key in before:self.assertEqual(result[key],before[key])


class PairedFeatureContracts(unittest.TestCase):
    def test_invalid_or_uncovered_reference_abstains(self):
        with self.assertRaises(ValueError):features.expected_in_source(np.zeros((20,20,3),np.uint8),np.zeros((3,3)),(20,20))
        self.assertEqual(features.valid_boxes([[0,0,10,10]],np.zeros((100,100),bool)),[])

    def test_pair_shapes_and_nonfinite_are_rejected(self):
        x=torch.zeros((2,1536));self.assertEqual(tuple(features.paired_features(x,x).shape),(2,6144))
        with self.assertRaises(ValueError):features.paired_features(x,torch.zeros((1,1536)))
        x[0,0]=float('nan')
        with self.assertRaises(ValueError):features.paired_features(x,x)

    def test_proposals_require_two_distinct_checkpoint_votes(self):
        row=dict(class_id=0,box_xyxy=[100,100,150,150],confidence=.1)
        def model(digest):return dict(source_sha256='same',weight_sha256=digest,
                predictions=dict(source_shape=[300,400],merged_predictions=[row]))
        self.assertFalse(features.proposals(model('a'),[model('a'),model('a')]))
        self.assertEqual(len(features.proposals(model('a'),[model('a'),model('b')])),1)

    def test_fixed_gate_budget_and_old_prefix(self):
        current=dict(primary=[],all_predictions=[])
        row=dict(class_id=0,box_xyxy=[100,100,150,150],confidence=.1,semantic_model_vote_sha256=['a','b'])
        before=copy.deepcopy(row)
        self.assertFalse(features.select(current,[row],[[.011,.979,.01]],'head')['paired_semantic_additions'])
        result=features.select(current,[row],[[.01,.98,.01]],'head')
        self.assertEqual(len(result['paired_semantic_additions']),1);self.assertEqual(row,before)
        old=dict(primary=[row],all_predictions=[row])
        self.assertEqual(features.select(old,[row],[[.01,.98,.01]],'head')['all_predictions'],[row])
        for scores in ([[float('nan'),.99,.01]],[[-.01,.99,.02]],[[.5,.99,.01]]):
            with self.assertRaises(ValueError):features.select(current,[row],scores,'head')


if __name__=='__main__':unittest.main()
