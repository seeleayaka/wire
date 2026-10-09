"""Synthetic pixels/fake model only: exercise real tiling/projection/edge code."""
import unittest
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import paired_common_scale640 as views
from inspection_agent.port_tiling import tile_windows


class SyntheticPredictor:
    def __init__(self, box=None, fail=False):
        self.box = box; self.fail = fail; self.calls = []

    def predict(self, crops, **kwargs):
        self.calls.append(([(p.shape[1], p.shape[0]) for p in crops], kwargs))
        if self.fail: raise RuntimeError('synthetic model failure')
        boxes = [] if self.box is None else [SimpleNamespace(xyxy=np.array([self.box], dtype=float),
                   conf=np.float64(.9), cls=np.int64(0))]
        return [SimpleNamespace(boxes=boxes) for _ in crops]


class ActualTileContractTests(unittest.TestCase):
    def test_real_tile_calls_keep_all_non_scale_model_options(self):
        before = deepcopy(views.CONFIG); model = SyntheticPredictor([200, 200, 240, 240])
        result = views.infer(model, np.zeros((1000, 1000, 3), dtype=np.uint8))
        self.assertEqual(result['windows'], [list(w) for w in tile_windows(1000, 1000, 640, 480)])
        self.assertEqual(result['raw_predictions'], 4); self.assertEqual(result['edge_rejected'], 0)
        self.assertEqual(sorted(r['box_xyxy'] for r in result['merged_predictions']),
                         [[200., 200., 240., 240.], [200., 560., 240., 600.],
                          [560., 200., 600., 240.], [560., 560., 600., 600.]])
        for dimensions, kwargs in model.calls:
            self.assertEqual(dimensions, [(640, 640)]*len(dimensions))
            self.assertEqual(kwargs, dict(imgsz=960, conf=before['predict_conf_floor'], iou=before['predict_iou'],
                             max_det=before['predict_max_det'], device='cpu', verbose=False, save=False))
        self.assertEqual(views.CONFIG, before)

    def test_real_native_boundaries_kept_artificial_cut_edges_rejected(self):
        result = views.infer(SyntheticPredictor([1, 1, 20, 20]), np.zeros((1000, 1000, 3), dtype=np.uint8))
        self.assertEqual((result['raw_predictions'], result['edge_rejected']), (4, 3))
        self.assertEqual(len(result['edge_kept_predictions']), 1)
        self.assertEqual(result['edge_kept_predictions'][0]['source_tile'], 0)

    def test_non_square_small_frame_preserves_original_coordinates(self):
        model = SyntheticPredictor([50, 10, 100, 30])
        result = views.infer(model, np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertEqual(result['windows'], [[0, 0, 200, 100]])
        self.assertEqual(result['source_shape'], [100, 200])
        self.assertEqual(result['merged_predictions'][0]['box_xyxy'], [50., 10., 100., 30.])
        self.assertEqual(model.calls[0][0], [(200, 100)])

    def test_actual_empty_prediction_records_windows_not_placeholder(self):
        result = views.infer(SyntheticPredictor(), np.zeros((700, 900, 3), dtype=np.uint8))
        self.assertTrue(result['windows'])
        self.assertEqual(result['raw_predictions'], 0)
        self.assertEqual(result['merged_predictions'], [])

    def test_real_model_exception_restores_frozen_options(self):
        before = deepcopy(views.CONFIG)
        with self.assertRaises(RuntimeError):
            views.infer(SyntheticPredictor(fail=True), np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertEqual(views.CONFIG, before)


if __name__ == '__main__': unittest.main(verbosity=2)
