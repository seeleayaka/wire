import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
import train_dense_quality as adapter


class QualityAdapterTests(unittest.TestCase):
    def test_history_list_passthrough_and_dictionary_provenance(self):
        runner=adapter.runner;calls=[]
        def sink(path,value):calls.append((path,value))
        def exercise():
            self.assertEqual(runner.BASE.name,'dense_quality_20261003')
            runner.save(Path('history.json'),[dict(epoch=1,loss=.2)])
            runner.save(Path('protocol.json'),dict(pins={},only_difference_from_D4_continuation='old marker'))
            runner.save(Path('report.json'),dict(status='complete'))
        with patch.object(runner,'save',side_effect=sink),patch.object(runner,'main',side_effect=exercise):adapter.main()
        self.assertEqual(calls[0][1],[dict(epoch=1,loss=.2)])
        self.assertTrue(calls[1][1]['quality_targets_training_only'])
        self.assertNotIn('only_difference_from_D4_continuation',calls[1][1])
        self.assertTrue(calls[2][1]['joint_localization_quality_training'])
    def test_runner_failure_restores_globals(self):
        runner=adapter.runner;before=(runner.BASE,runner.loss,runner.save)
        with patch.object(runner,'main',side_effect=ValueError('synthetic')):
            with self.assertRaises(ValueError):adapter.main()
        self.assertEqual((runner.BASE,runner.loss,runner.save),before)


if __name__=='__main__':unittest.main()
