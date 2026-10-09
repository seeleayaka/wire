import unittest
from copy import deepcopy
from unittest.mock import patch
import paired_common_scale640 as views
from inspection_agent.port_tiling import tile_windows


class CommonScaleTests(unittest.TestCase):
    def test_only_physical_scale_changes_and_restores(self):
        before=deepcopy(views.CONFIG)
        def stub(model,image):
            expected=deepcopy(before);expected.update(tile_size=640,tile_stride=480)
            self.assertEqual(views.CONFIG,expected)
            self.assertEqual(model,'model');self.assertEqual(image,'image')
            return {'actual_stub_not_inference':True}
        with patch.object(views,'predict',side_effect=stub):
            self.assertEqual(views.infer('model','image'),{'actual_stub_not_inference':True})
        self.assertEqual(views.CONFIG,before)

    def test_failure_restores_every_original_gate(self):
        before=deepcopy(views.CONFIG)
        with patch.object(views,'predict',side_effect=RuntimeError('controlled fixture failure')):
            with self.assertRaises(RuntimeError):views.infer(None,None)
        self.assertEqual(views.CONFIG,before)

    def test_changed_model_resolution_rejected_without_inference(self):
        with patch.dict(views.CONFIG,{'predict_imgsz':640}),patch.object(views,'predict') as predict:
            with self.assertRaises(ValueError):views.infer(None,None)
            predict.assert_not_called()

    def test_actual_full_frame_window_grid_has48_windows(self):
        grid=tile_windows(3648,2736,640,480)
        self.assertEqual(len(grid),48)
        self.assertEqual(len(set(tuple(w) for w in grid)),48)
        for l,t,r,b in grid:self.assertTrue(0<=l<r<=3648 and 0<=t<b<=2736)


if __name__=='__main__':unittest.main(verbosity=2)
