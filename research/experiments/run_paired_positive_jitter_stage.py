"""Reuse fixed original gates with central-context, not footprint, features."""
import argparse
import sys
sys.dont_write_bytecode = True
import run_paired_boxpool_stage as gates
import evaluate_paired_boxpool_full_train as final_gate
from prepare_paired_positive_jitter import OUT
from port_semantic_verifier import embeddings


def execute(stage):
    old_out, old_embeddings = gates.OUT, gates.embeddings
    try:
        gates.OUT, gates.embeddings = OUT, embeddings
        return gates.execute(stage)
    finally:
        gates.OUT, gates.embeddings = old_out, old_embeddings


def exact_full_train():
    old_out = final_gate.OUT
    try:
        final_gate.OUT = OUT
        return final_gate.main()
    finally:
        final_gate.OUT = old_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=tuple(gates.MODULES) + ('exact_full_train',))
    stage = parser.parse_args().stage
    exact_full_train() if stage == 'exact_full_train' else execute(stage)
