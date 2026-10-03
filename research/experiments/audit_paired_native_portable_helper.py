"""All270 GT-free proposals and all462 scored cases, staged helper parity."""
import importlib.util
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
from current_port_baseline_audit import BASE,read_current_case
from paired_pose_search import proposals as experimental_proposals
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'
INPUTS=ROOT/'artifacts/paired_pose_search_inputs_20261003'
HELPER=ROOT/'staging/paired_native_pose_release/paired_native_pose_features.py'
OUT=ROOT/'artifacts/paired_native_portable_parity_20261004'


def main():
    if OUT.exists():raise FileExistsError('Preserve staged helper replay')
    spec=importlib.util.spec_from_file_location('staged_native_features',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    pins={str(p):sha(p) for p in (Path(__file__),HELPER,SOURCE/'report.json')};proposal_counts={}
    for stage in ('train','inner','outer'):
        indexpath=PREP/stage/'index.json';pins[str(indexpath)]=sha(indexpath);indexed={r['image']:r for r in load(indexpath)['records']}
        count=0
        for entry in load(BASE/stage/'report.json')['cases']:
            name=entry['image'];path=Path(indexed[name]['path']);assert sha(path)==indexed[name]['sha256'];pins[str(path)]=sha(path)
            case=load(path);teacher,old=read_current_case(stage,entry,pins);assert teacher==case['teacher']
            prepared=INPUTS/(stage+'_'+Path(name).stem+'_proposals.json');pins[str(prepared)]=sha(prepared);source=load(prepared)
            models=[teacher,case['student'],case['feature'],old['alternative']];current=source['current']
            native=helper.proposals(teacher,models,current) if source['remaining_slots'] else []
            repeat=experimental_proposals(teacher,models,current) if source['remaining_slots'] else []
            assert native==repeat==source['extended_candidates'],(stage,name)
            count+=len(native)
        proposal_counts[stage]=count
    selection_counts={}
    for stage in load(SOURCE/'report.json')['stages']:
        dataset='train' if stage in ('source_train_oof','full_train') else stage;count=0
        for entry in load(BASE/dataset/'report.json')['cases']:
            path=SOURCE/stage/(Path(entry['image']).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            repeated=helper.select(case['current'],case['proposals'],case['probabilities'],case['head_sha256'])
            assert repeated==case['trial'],(stage,entry['image']);count+=1
        selection_counts[stage]=count
    assert sum(selection_counts.values())==462 and {p:sha(Path(p)) for p in pins}==pins
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',proposal_counts=proposal_counts,selection_counts=selection_counts,
        total_source_cases=270,total_selection_cases=462,pins=pins,all_proposal_and_selection_dictionaries_equal=True,
        installed=False,field_accuracy=False))
    print(str(dict(proposal_counts=proposal_counts,selection_counts=selection_counts)),flush=True)


if __name__=='__main__':main()
