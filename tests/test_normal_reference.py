from __future__ import annotations

import unittest

import numpy as np

from inspection_agent.normal_reference import (
    classification_metrics,
    descriptor_from_patch_features,
    nearest_normal_score,
    select_balanced_threshold,
)


class NormalReferenceTests(unittest.TestCase):
    def test_spatial_descriptor_is_normalized_and_retains_quadrants(self) -> None:
        features = np.zeros((4, 4, 2), dtype=np.float32)
        features[:2, :2, 0] = 1.0
        features[2:, 2:, 1] = 1.0
        descriptor = descriptor_from_patch_features(features, grid_size=2)
        self.assertEqual(descriptor.shape, (10,))
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=6)
        self.assertGreater(descriptor[2], descriptor[3])
        self.assertGreater(descriptor[-1], descriptor[-2])

    def test_nearest_normal_score_uses_k_closest_cosine_distances(self) -> None:
        bank = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        score, neighbors = nearest_normal_score(query, bank, k=2)
        self.assertAlmostEqual(score, 0.1, places=6)
        self.assertEqual([item["index"] for item in neighbors], [0, 1])

    def test_threshold_selection_honors_sensitivity_floor(self) -> None:
        scores = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]
        labels = [False, False, False, True, True, True]
        selection = select_balanced_threshold(scores, labels, minimum_sensitivity=0.90)
        self.assertGreater(selection["threshold"], 0.30)
        self.assertLessEqual(selection["threshold"], 0.70)
        self.assertEqual(selection["metrics"]["balanced_accuracy"], 1.0)

    def test_metrics_report_binary_confusion_counts(self) -> None:
        metrics = classification_metrics(
            scores=[0.1, 0.8, 0.7, 0.2],
            labels=[False, True, False, True],
            threshold=0.5,
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
