"""Scoped original gate reuse, with new features and accepted-paired baseline."""
import argparse,copy,importlib,sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_boxpool_features import OUT
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from prepare_paired_port_semantics import ROOT,load,save,sha
from paired_boxpool_features import embeddings
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
BASELINE=ROOT/'artifacts/paired_boxpool_current_baseline_20261003'
MODULES=dict(classifier='train_paired_port_semantic_heads',source='evaluate_paired_port_semantic_train',
             full='train_paired_port_semantic_full',holdouts='evaluate_paired_port_semantic_holdouts')


def prepare_baseline():
    from current_port_baseline_audit import BASE
    if BASELINE.exists():
        record=load(BASELINE/'protocol.json');assert {p:sha(Path(p)) for p in record['pins']}==record['pins'];return
    BASELINE.mkdir();pins={}
    for stage in ('train','inner','outer'):
        path=BASE/stage/'report.json';pins[str(path)]=sha(path);report=copy.deepcopy(load(path))
        accepted_path=(GEOMETRY/'full_train' if stage=='train' else GEOMETRY/'holdouts'/stage)/'report.json'
        pins[str(accepted_path)]=sha(accepted_path);accepted=load(accepted_path)
        by_name={r['image']:r for r in accepted['cases']}
        for row in report['cases']:row['trial']=copy.deepcopy(by_name[row['image']]['trial'])
        report['summary']['trial']=copy.deepcopy(accepted['summary']['trial'])
        directory=BASELINE/stage;directory.mkdir();save(directory/'report.json',report)
    save(BASELINE/'protocol.json',dict(pins=pins,source_baseline='accepted_paired_geometry_fixed_full_head',
        counts=[286,64,33],unmatched=[4,0,1],no_GT_selection=True,no_deployment=True))


def execute(stage):
    module=importlib.import_module(MODULES[stage]);old_out,old_load=module.OUT,module.load
    old_base=getattr(module,'BASE',None);old_embeddings=getattr(module,'embeddings',None)
    frozen=paired_runtime_fingerprint('E:/PythonProject10');prepare_baseline()
    def adapted_load(path):
        payload=old_load(path)
        if stage in ('source','holdouts') and str(path).startswith(str(module.PROPOSALS)) and Path(path).name.endswith('_proposals.json'):
            name=payload['image'];split=Path(path).name.split('_',1)[0]
            fixed=(GEOMETRY/'full_train' if split=='train' else GEOMETRY/'holdouts'/split)/(Path(name).stem+'_predictions.json')
            payload=copy.deepcopy(payload);payload['current']=copy.deepcopy(old_load(fixed)['trial'])
        return payload
    try:
        module.OUT=OUT;module.load=adapted_load
        if old_base is not None:module.BASE=BASELINE
        if old_embeddings is not None:module.embeddings=embeddings
        module.main();assert paired_runtime_fingerprint('E:/PythonProject10')==frozen
    finally:
        module.OUT,module.load=old_out,old_load
        if old_base is not None:module.BASE=old_base
        if old_embeddings is not None:module.embeddings=old_embeddings

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=tuple(MODULES));execute(parser.parse_args().stage)
