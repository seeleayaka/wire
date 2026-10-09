import unittest
from unittest.mock import patch
import numpy as np
from audit_harness_reference_trial import eligible_components
import run_harness_reference_trial as runner
from replay_harness_reference_trial import parts


class HarnessReferenceTests(unittest.TestCase):
    def setUp(self):
        self.anchors = [{'id': 'a', 'bbox_xyxy': [2, 2, 4, 4]},
                        {'id': 'b', 'bbox_xyxy': [12, 2, 14, 4]}]

    def test_native_continuous_component(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3, 3:14] = 255
        before = mask.copy()
        good, _ = eligible_components(mask, .9, self.anchors, [0, 0])
        self.assertEqual(good, [1])
        np.testing.assert_array_equal(before, mask)

    def test_gap_never_bridged(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3, 3:14] = 255
        mask[3, 8] = 0
        good, parts = eligible_components(mask, .9, self.anchors, [0, 0])
        self.assertEqual(good, [])
        self.assertEqual(len(parts), 2)

    def test_low_score_not_promoted(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3, 3:14] = 255
        self.assertEqual(eligible_components(mask, .749, self.anchors, [0, 0])[0], [])

    def test_boundary_guard_preserved(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3, :14] = 255
        self.assertEqual(eligible_components(mask, .9, self.anchors, [0, 0])[0], [])

    def test_active_job_blocks_new_inference_before_any_writes(self):
        with patch.object(runner, 'read', return_value={'status': 'running'}):
            with self.assertRaisesRegex(ValueError, 'protect active'):
                runner.prepare()

    def test_independent_flood_fill_preserves_gap_and_diagonal(self):
        mask = np.zeros((8, 8), bool)
        mask[2, 2] = mask[3, 3] = mask[5, 5] = True
        before = mask.copy()
        self.assertEqual(sorted(len(p) for p in parts(mask)), [1, 2])
        np.testing.assert_array_equal(mask, before)


if __name__ == '__main__':
    unittest.main()
