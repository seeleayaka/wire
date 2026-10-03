import copy
import unittest
from unittest.mock import patch
import paired_pose_final_gate_backend as module


class FinalGateAdapter(unittest.TestCase):
    def test_restore_patches_even_on_backend_failure(self):
        old = module.backend.proposals,module.backend.select
        original = dict(median_geometry_evidence=dict(native=dict(primary=[],all_predictions=[])))
        with patch.object(module,'median_runtime_fingerprint',return_value={}),patch.object(module.backend,'append_median_review',side_effect=RuntimeError('synthetic')):
            with self.assertRaises(RuntimeError): module.append_pose_review({},original,project='ignored')
        self.assertEqual(old,(module.backend.proposals,module.backend.select))
    def test_original_median_fields_and_native_prefix_preserved(self):
        row = dict(class_id=0,confidence=.8,box_xyxy=[100,100,150,140])
        original = dict(median_geometry_policy=dict(enabled=True,policy_id='accepted'),
            median_geometry_evidence=dict(native=dict(primary=[row],all_predictions=[row])),supplementary_hints=[])
        before = copy.deepcopy(original)
        def fake(report,current,*,project):
            selected = module.backend.select(dict(primary=[],all_predictions=[]),[],[],'head')
            result = copy.deepcopy(current); result['median_geometry_policy'] = dict(added_hints=0)
            result['median_geometry_evidence'] = dict(native=selected); return result
        with patch.object(module,'median_runtime_fingerprint',return_value={}),patch.object(module.backend,'append_median_review',side_effect=fake):
            output = module.append_pose_review({},original,project='ignored')
        self.assertEqual(original,before)
        self.assertEqual(output['median_geometry_evidence'],before['median_geometry_evidence'])
        self.assertEqual(output['pose_geometry_evidence']['native']['all_predictions'],[row])


if __name__ == '__main__': unittest.main()
