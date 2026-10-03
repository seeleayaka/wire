import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import run_mendeley_cnn_experimental_review as entry


class EntryTests(unittest.TestCase):
    def fake(self, localize=False, failure=False):
        tiled = SimpleNamespace(_merge_candidates=lambda rows: rows, LARGE_ROI_MAX_CANDIDATES=9)
        impl = SimpleNamespace(review_components=lambda *a, **k: None)
        cascade = SimpleNamespace(dino_fused_regions=lambda *a: ('overlay', 'heat', [{'left': 1}]))
        original = cascade.dino_fused_regions
        def review(*args, **options):
            tiled.LARGE_ROI_MAX_CANDIDATES = 6
            if failure: raise RuntimeError('old pipeline failure')
            candidates = cascade.dino_fused_regions(None, None, None)[2] if localize else []
            return {'decision': 'manual' if localize else 'normal', 'candidates': candidates}
        cascade.review_image = review
        return cascade, impl, tiled, original

    def test_disabled_delegates_without_setup(self):
        cascade, _, _, _ = self.fake(True)
        expected = copy.deepcopy(cascade.review_image(None, None, None))
        with patch('tools.merge_audit.setup', side_effect=AssertionError('must not load CNN')):
            report = entry.review_image(None, None, None, cascade=cascade)
        self.assertEqual(report.pop('experimental_cnn')['status'], 'disabled')
        self.assertEqual(report, expected)

    def test_upstream_normal_skips_evidence_and_restores_hooks(self):
        cascade, impl, tiled, original = self.fake()
        merge, review = tiled._merge_candidates, impl.review_components
        with patch('tools.merge_audit.setup', return_value=(impl, tiled)), patch('inspection_agent.experimental_cnn.FrozenCnnEvidence', side_effect=AssertionError()):
            report = entry.review_image(None, None, None, cascade=cascade, experimental_cnn=True)
        self.assertEqual(report['experimental_cnn']['status'], 'skipped_upstream')
        self.assertIs(cascade.dino_fused_regions, original)
        self.assertIs(tiled._merge_candidates, merge)
        self.assertIs(impl.review_components, review)
        self.assertEqual(tiled.LARGE_ROI_MAX_CANDIDATES, 9)

    def test_capture_failure_keeps_old_candidates(self):
        cascade, impl, tiled, original = self.fake(True)
        with patch('tools.merge_audit.setup', return_value=(impl, tiled)):
            report = entry.review_image(None, None, None, cascade=cascade, experimental_cnn=True)
        self.assertEqual(report['candidates'], [{'left': 1}])
        self.assertEqual(report['experimental_cnn']['status'], 'fallback')
        self.assertIs(cascade.dino_fused_regions, original)

    def test_old_pipeline_errors_propagate_hooks_restored(self):
        cascade, impl, tiled, original = self.fake(failure=True)
        with patch('tools.merge_audit.setup', return_value=(impl, tiled)):
            with self.assertRaisesRegex(RuntimeError, 'old pipeline failure'):
                entry.review_image(None, None, None, cascade=cascade, experimental_cnn=True)
        self.assertIs(cascade.dino_fused_regions, original)
        self.assertEqual(tiled.LARGE_ROI_MAX_CANDIDATES, 9)


if __name__ == '__main__': unittest.main()
