import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paired_positive_jitter_stage as stage


def check_central_embeddings_and_output_are_scoped_even_on_failure():
    original = stage.gates.OUT, stage.gates.embeddings
    def fail(name):
        assert name == 'holdouts'
        assert stage.gates.OUT == stage.OUT
        assert stage.gates.embeddings is stage.embeddings
        raise RuntimeError('intentional test failure')
    with patch.object(stage.gates, 'execute', side_effect=fail):
        try:
            stage.execute('holdouts')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Failure must propagate')
    assert (stage.gates.OUT, stage.gates.embeddings) == original


def check_final_head_output_restored():
    original = stage.final_gate.OUT
    def check():
        assert stage.final_gate.OUT == stage.OUT
        return 'checked'
    with patch.object(stage.final_gate, 'main', side_effect=check):
        assert stage.exact_full_train() == 'checked'
    assert stage.final_gate.OUT == original


class ScopedJitterStages(unittest.TestCase):
    def test_central_embeddings_and_output_are_scoped_even_on_failure(self):
        check_central_embeddings_and_output_are_scoped_even_on_failure()

    def test_final_head_output_restored(self):
        check_final_head_output_restored()


if __name__ == '__main__':
    unittest.main()
