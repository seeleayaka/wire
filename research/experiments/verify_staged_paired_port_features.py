"""ALL270 formalization parity, no GT or metric change and no installation."""
import importlib.util
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
from port_semantic_model_vote import proposals
from paired_port_semantic_selection import select
feature_path=ROOT/'staging/paired_geometry_release/paired_port_features.py'
spec=importlib.util.spec_from_file_location('staged_paired_features',feature_path)
staged=importlib.util.module_from_spec(spec);spec.loader.exec_module(staged)


def main():
    out=ROOT/'artifacts/paired_geometry_formalization_parity_20261003'
    if out.exists():raise FileExistsError('Preserve parity evidence')
    detectors=ROOT/'artifacts/paired_support_graph_20261003/source_selections'
    trained=ROOT/'artifacts/paired_semantic_localization_20261003';pins={str(feature_path):sha(feature_path),str(Path(__file__)):sha(Path(__file__))}
    rows=[];totals={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        indexpath=detectors/stage/'index.json';pins[str(indexpath)]=sha(indexpath);index=load(indexpath)
        assert len(index['records'])==count
        for entry in index['records']:
            path=Path(entry['path']);assert sha(path)==entry['sha256'];pins[str(path)]=sha(path);case=load(path)
            first=proposals(case['teacher'],[case['teacher'],case['student']])
            second=staged.proposals(case['teacher'],[case['teacher'],case['student']]);assert first==second
            path=(trained/'full_train' if stage=='train' else trained/'holdouts'/stage)/(Path(entry['image']).stem+'_predictions.json')
            pins[str(path)]=sha(path);prediction=load(path)
            left=select(prediction['current'],prediction['proposals'],prediction['probabilities'],prediction['head_sha256'])
            right=staged.select(prediction['current'],prediction['proposals'],prediction['probabilities'],prediction['head_sha256'])
            assert left==right==prediction['trial']
            rows.append(dict(stage=stage,image=entry['image'],weak_pool=len(first),additions=len(right['paired_semantic_additions'])))
        totals[stage]=sum(row['additions'] for row in rows if row['stage']==stage)
    assert {p:sha(Path(p)) for p in pins}==pins
    out.mkdir();save(out/'report.json',dict(status='complete',all270_proposal_and_selector_dictionary_parity=True,
        additions=totals,cases=rows,pins=pins,preprocessing_primitive_parity_unit_tested=True,
        not_new_accuracy=True,no_installation=True,field_accuracy=False))
    print(str(dict(status='complete',sources=len(rows),additions=totals)))


if __name__=='__main__':main()
