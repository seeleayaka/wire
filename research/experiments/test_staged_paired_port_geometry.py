import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=['E:/PythonProject10','E:/PythonProject10/prototype']
folder=Path(__file__).resolve().parents[1]/'staging/paired_geometry_release'


def load_staged():
    spec=importlib.util.spec_from_file_location('inspection_agent.paired_port_features',folder/'paired_port_features.py')
    features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)
    with patch.dict(sys.modules,{'inspection_agent.paired_port_features':features}):
        spec=importlib.util.spec_from_file_location('staged_geometry_backend',folder/'paired_port_geometry.py')
        backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    return backend


backend=load_staged()


def original():
    return dict(status='applied',parents=[],existing_hints=[],rescue_hints=[],supplementary_hints=[{'box':{'old':True}}],
                decision='possible_difference_manual_review')


class FormalBackendGuards(unittest.TestCase):
    def test_new_switch_defaults_off_without_model_or_manifest_reads(self):
        old=original();before=copy.deepcopy(old)
        with patch.object(backend,'run_resolution_plug_review',return_value=old):
            with patch.object(backend,'paired_runtime_fingerprint',side_effect=AssertionError('No new reads')):
                result=backend.run_paired_geometry_review({},project='unused')
        self.assertFalse(result['paired_geometry_policy']['enabled'])
        for key in before:self.assertEqual(result[key],before[key])

    def test_changed_manifest_preserves_current_cues(self):
        old=original();before=copy.deepcopy(old)
        frozen=dict(manifest='wrong',head=backend.HEAD_SHA,encoder=backend.ENCODER_SHA)
        with patch.object(backend,'run_resolution_plug_review',return_value=old):
            with patch.object(backend,'paired_runtime_fingerprint',return_value=frozen):
                result=backend.run_paired_geometry_review({},project='unused',paired_enabled=True)
        self.assertIn('paired_model_provenance_mismatch',result['paired_geometry_policy']['fallback_reason'])
        for key in before:self.assertEqual(result[key],before[key])

    def test_full_current_budget_never_allocates_extra_model(self):
        old=original();old['supplementary_hints']*=5
        with patch.object(backend,'run_resolution_plug_review',return_value=old):
            with patch.object(backend,'paired_runtime_fingerprint',side_effect=AssertionError('No new reads')):
                result=backend.run_paired_geometry_review({},project='unused',paired_enabled=True)
        self.assertEqual(len(result['supplementary_hints']),5)
        self.assertEqual(result['paired_geometry_policy']['added_hints'],0)


if __name__=='__main__':unittest.main()
