import unittest
from pathlib import Path
from unittest.mock import patch
import run_paired_rectcontext_stage as stage
import run_paired_rectcontext_pipeline as pipeline


class ScopedRectangleStages(unittest.TestCase):
    def test_original_embeddings_and_stronger_prefix_restored_on_error(self):
        class Module:
            OUT = Path('old')
            BASE = Path('old_baseline')
            PROPOSALS = Path('proposals')
            embeddings = lambda *a: 'old'
            load = staticmethod(lambda path: {})
            def main(self):
                assert self.OUT == stage.OUT and self.BASE == stage.BASELINE
                assert self.embeddings is stage.embeddings
                raise RuntimeError('intentional')
        module = Module(); previous = module.OUT, module.BASE, module.embeddings, module.load
        with patch.object(stage, 'prepare_baseline'), patch.object(stage.importlib, 'import_module', return_value=module), patch.object(stage, 'paired_runtime_fingerprint', return_value={}):
            with self.assertRaises(RuntimeError): stage.execute('holdouts')
        self.assertEqual((module.OUT, module.BASE, module.embeddings, module.load), previous)

    def test_pipeline_scope_restored_when_crop_rejects(self):
        previous = pipeline.gates.OUT, pipeline.full_gate.OUT
        destination = Path('test_control')
        with patch.object(pipeline, 'progress'), patch.object(pipeline.gates, 'execute'), patch.object(pipeline, 'load', return_value={'qualifies_crop_feasibility': False}):
            self.assertEqual(pipeline.run_gates(destination), 'crop_rejected')
        self.assertEqual((pipeline.gates.OUT, pipeline.full_gate.OUT), previous)


if __name__ == '__main__': unittest.main()
