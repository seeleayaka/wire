import unittest
from unittest.mock import patch
import run_paired_pose_search as stage


class PoseStage(unittest.TestCase):
    def test_scoped_adapter_restores_on_failure(self):
        old = (stage.gate.OUT, stage.gate.PROPOSALS, stage.gate.load, stage.gate.save, stage.gate.select)
        def fail():
            self.assertIs(stage.gate.select, stage.select)
            self.assertEqual(stage.gate.OUT, stage.OUT)
            raise RuntimeError('synthetic failure before any expensive inference')
        with patch.object(stage.Path, 'exists', return_value=False), patch.object(stage, 'prepare_inputs', return_value={}), patch.object(stage, 'median_runtime_fingerprint', return_value={'fixed': True}), patch.object(stage.gate, 'main', side_effect=fail):
            with self.assertRaises(RuntimeError): stage.execute()
        self.assertEqual(old, (stage.gate.OUT, stage.gate.PROPOSALS, stage.gate.load, stage.gate.save, stage.gate.select))


if __name__ == '__main__': unittest.main()
