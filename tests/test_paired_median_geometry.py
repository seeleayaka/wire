import copy
import unittest
from unittest.mock import patch
from inspection_agent import paired_median_geometry as backend
from inspection_agent.paired_port_median_features import proposals


def model(weight, box, score=.6, source='source'):
    return dict(weight_sha256=weight, source_sha256=source,
        predictions=dict(source_shape=[200, 200], merged_predictions=[dict(class_id=0, box_xyxy=box, confidence=score)]))


class PortableMedianGeometry(unittest.TestCase):
    def test_default_off_keeps_upstream(self):
        original = dict(status='applied', supplementary_hints=[], rescue_hints=[])
        with patch.object(backend, 'run_paired_geometry_review', return_value=copy.deepcopy(original)) as upstream:
            output = backend.run_paired_median_review({}, project='E:/PythonProject10')
        upstream.assert_called_once_with({}, project='E:/PythonProject10', paired_enabled=False)
        for name, value in original.items(): self.assertEqual(output[name], value)
        self.assertFalse(output['median_geometry_policy']['enabled'])
        self.assertEqual(output['median_geometry_policy']['added_hints'], 0)

    def test_fallback_original_is_preserved(self):
        original = dict(status='fallback', fallback_reason='local_alignment_not_supported', supplementary_hints=[])
        before = copy.deepcopy(original)
        output = backend.append_median_review({}, original, project='E:/PythonProject10')
        self.assertEqual(original, before)
        for key, value in before.items(): self.assertEqual(output[key], value)
        self.assertEqual(output['median_geometry_policy']['added_hints'], 0)

    def test_full_budget_is_not_replaced(self):
        original = dict(status='applied', supplementary_hints=[dict(index=i) for i in range(5)])
        output = backend.append_median_review({}, original, project='E:/PythonProject10')
        self.assertEqual(output['supplementary_hints'], original['supplementary_hints'])
        self.assertEqual(output['median_geometry_policy']['added_hints'], 0)

    def test_bad_manifest_returns_old_result(self):
        original = dict(status='applied', supplementary_hints=[], rescue_hints=[])
        with patch.object(backend, 'median_runtime_fingerprint', return_value=dict(manifest='changed')):
            output = backend.append_median_review({}, original, project='E:/PythonProject10')
        self.assertIn('median_manifest_identity_mismatch', output['median_geometry_policy']['fallback_reason'])
        for key, value in original.items(): self.assertEqual(output[key], value)

    def test_fingerprint_binds_new_geometry_and_old_head(self):
        with patch.object(backend, 'paired_runtime_fingerprint', return_value={'old': 'pinned'}), patch.object(backend, 'sha', side_effect=lambda p: str(p)):
            value = backend.median_runtime_fingerprint('E:/PythonProject10')
        self.assertEqual(value['accepted'], {'old': 'pinned'})
        self.assertIn('paired_port_median_features.py', value['geometry'])
        self.assertIn('last_head.pt', value['head'])
        self.assertIn('paired_median_geometry_20261003.json', value['manifest'])

    def test_same_checkpoint_views_are_not_two_votes(self):
        a = model('a', [30, 40, 70, 60], .9); b = model('a', [32, 40, 72, 60], .8)
        self.assertEqual(proposals(a, [a, b]), [])
        c = model('c', [34, 40, 74, 60])
        value = proposals(a, [a, b, c])[0]
        self.assertEqual(value['box_xyxy'], [32., 40., 72., 60.])
        self.assertEqual(value['median_unique_checkpoint_count'], 2)

    def test_source_mismatch_rejected(self):
        a = model('a', [30, 40, 70, 60])
        with self.assertRaises(ValueError): proposals(a, [a, model('b', [30, 40, 70, 60], source='stale')])

    def test_deterministic_order_and_no_mutation(self):
        a = model('a', [30, 40, 70, 60]); b = model('b', [34, 40, 74, 60]); before = copy.deepcopy([a, b])
        self.assertEqual(proposals(a, [a, b]), proposals(a, [b, a]))
        self.assertEqual([a, b], before)


if __name__ == '__main__': unittest.main()
