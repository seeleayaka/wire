import copy
import unittest
from unittest.mock import patch
import paired_fine_tile_views as module


class FineTiles(unittest.TestCase):
    def test_config_restored_even_on_inference_failure(self):
        old = copy.deepcopy(module.CONFIG)
        def fail(model,image):
            self.assertEqual(module.CONFIG['tile_size'],960)
            self.assertEqual(module.CONFIG['tile_stride'],720)
            self.assertEqual(module.CONFIG['predict_imgsz'],old['predict_imgsz'])
            self.assertEqual(module.CONFIG['predict_conf_floor'],old['predict_conf_floor'])
            raise RuntimeError('synthetic failure')
        with patch.object(module,'predict',side_effect=fail):
            with self.assertRaises(RuntimeError): module.infer(None,None)
        self.assertEqual(module.CONFIG,old)


if __name__ == '__main__': unittest.main()
