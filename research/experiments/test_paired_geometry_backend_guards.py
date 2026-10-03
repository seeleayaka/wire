import copy
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,'E:/PythonProject10/prototype')
import paired_port_geometry_backend as backend


def baseline(status='applied',supplementary=None):
    return dict(status=status,parents=[dict(bbox_xyxy=[1,2,3,4])],existing_hints=[],
                rescue_hints=[],supplementary_hints=supplementary or [],decision='possible_difference_manual_review')


class BackendGuards(unittest.TestCase):
    def test_disabled_upstream_and_full_extra_budget_do_not_infer(self):
        for original in (baseline('disabled'),baseline(supplementary=[dict(box={'id':i}) for i in range(5)])):
            before=copy.deepcopy(original)
            with patch.object(backend,'load',side_effect=AssertionError('No model loading')):
                result=backend.append_geometry_review({},original,project='E:/PythonProject10')
            for key in before:self.assertEqual(result[key],before[key])
            self.assertEqual(original,before)
            self.assertEqual(result['paired_geometry_policy']['added_hints'],0)

    def test_runtime_drift_preserves_all_old_cues_and_reports_fallback(self):
        original=baseline(supplementary=[dict(box={'id':'old'})]);before=copy.deepcopy(original)
        with patch.object(backend,'resolution_runtime_fingerprint',return_value={'runtime':'changed'}):
            with patch.object(backend,'load',return_value={'runtime_fingerprint':{'runtime':'original'}}):
                result=backend.append_geometry_review({},original,project='E:/PythonProject10')
        self.assertIn('accepted_runtime_changed',result['paired_geometry_policy']['fallback_reason'])
        self.assertEqual(result['supplementary_hints'],before['supplementary_hints'])
        self.assertEqual(original,before)


if __name__=='__main__':unittest.main()
