import copy
import unittest
from paired_median_geometry_backend import append_median_review


class MedianSafety(unittest.TestCase):
    def test_original_fallback_stays_fallback_and_immutable(self):
        original = dict(status='fallback', fallback_reason='local_alignment_not_supported', supplementary_hints=[])
        protected = copy.deepcopy(original)
        result = append_median_review({}, original, project='E:/PythonProject10')
        self.assertEqual(original, protected)
        for name, value in original.items(): self.assertEqual(result[name], value)
        self.assertEqual(result['median_geometry_policy']['added_hints'], 0)
        self.assertNotIn('median_geometry_evidence', result)

    def test_full_budget_does_not_infer_or_replace(self):
        original = dict(status='applied', supplementary_hints=[dict(index=i) for i in range(5)])
        protected = copy.deepcopy(original)
        result = append_median_review({}, original, project='E:/PythonProject10')
        self.assertEqual(original, protected)
        self.assertEqual(result['supplementary_hints'], protected['supplementary_hints'])
        self.assertEqual(result['median_geometry_policy']['added_hints'], 0)


if __name__ == '__main__': unittest.main()
