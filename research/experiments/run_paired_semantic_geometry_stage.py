"""Scoped adapter preserves original gate implementations and restores globals."""
import argparse
import copy
import importlib
import sys
sys.dont_write_bytecode=True
from prepare_paired_semantic_geometry import NEW


MODULES=dict(classifier='train_paired_port_semantic_heads',source='evaluate_paired_port_semantic_train',
             full='train_paired_port_semantic_full',holdouts='evaluate_paired_port_semantic_holdouts')


def execute(stage):
    module=importlib.import_module(MODULES[stage]);old_out=module.OUT;old_load=module.load
    def load(path):
        payload=old_load(path)
        if stage=='source' and path==NEW/'features_train/report.json':
            payload=copy.deepcopy(payload)
            ordering={name:index%3 for index,name in enumerate(sorted(payload['source_groups']))}
            for row in payload['sources']:row['fold']=ordering[row['image']]
        return payload
    try:
        module.OUT=NEW;module.load=load;module.main()
    finally:
        module.OUT=old_out;module.load=old_load


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=tuple(MODULES));execute(parser.parse_args().stage)
