import types,unittest
from unittest.mock import patch
from pathlib import Path
import run_paired_boxpool_stage as adapter
from prepare_paired_port_semantics import PROPOSALS,load


class FootprintStageAdapter(unittest.TestCase):
    def test_accepted_baseline_override_and_globals_restore_after_failure(self):
        before_out=Path('unchanged');before_base=Path('old_baseline')
        module=types.SimpleNamespace(OUT=before_out,BASE=before_base,load=load,PROPOSALS=PROPOSALS)
        candidate=PROPOSALS/'train_disconnected_001_proposals.json'
        expected=load(adapter.GEOMETRY/'full_train/disconnected_001_predictions.json')['trial']
        def execute():
            self.assertEqual(module.load(candidate)['current'],expected)
            self.assertEqual(module.BASE,adapter.BASELINE)
            raise RuntimeError('intentional restoration test')
        module.main=execute
        with patch.object(adapter.importlib,'import_module',return_value=module):
            with self.assertRaisesRegex(RuntimeError,'intentional restoration test'):adapter.execute('source')
        self.assertEqual(module.OUT,before_out);self.assertEqual(module.BASE,before_base);self.assertIs(module.load,load)

if __name__=='__main__':unittest.main()
