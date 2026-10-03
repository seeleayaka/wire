"""Preserve reused evaluator; supply verified original alignment metadata only."""
import copy
import sys
sys.dont_write_bytecode=True
from prepare_paired_semantic_geometry import NEW
from prepare_paired_port_semantics import OUT as ORIGINAL,load,sha
import prepare_paired_semantic_full_train as evaluator


def main():
    old_out,old_new,old_load=evaluator.OUT,evaluator.NEW,evaluator.load
    original=load(ORIGINAL/'features_train/report.json')
    assert {p:sha(__import__('pathlib').Path(p)) for p in original['pins']}==original['pins']
    metadata={row['image']:row for row in original['sources']}
    def adapted(path):
        payload=old_load(path)
        if path==NEW/'features_train/report.json':
            payload=copy.deepcopy(payload)
            for row in payload['sources']:
                verified=metadata[row['image']]
                row.update(alignment=verified['alignment'],source_sha256=verified['source_sha256'])
        return payload
    try:evaluator.OUT=NEW;evaluator.NEW=NEW;evaluator.load=adapted;evaluator.main()
    finally:evaluator.OUT=old_out;evaluator.NEW=old_new;evaluator.load=old_load


if __name__=='__main__':main()
