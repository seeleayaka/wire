"""GT-free existing-checkpoint proposal extension, then post-hoc ceiling."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE,read_targets,read_current_case
from inspection_agent.paired_port_features import proposals
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
from inspection_agent.teacher_student_port_support import TEACHER_SHA,STUDENT_SHA
from inspection_agent.feature_residual_port_support import WEIGHT_SHA
from inspection_agent.port_tiling import box_iou
from audit_port_multiscale_acceptance import metric
from audit_port_matching_cardinality import maximum_matching
OUT=ROOT/'artifacts/paired_multimodel_proposal_ceiling_20261003'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'


def main():
    if OUT.exists():raise FileExistsError('Preserve proposal ceilings')
    frozen=paired_runtime_fingerprint(REPO);pins={str(p):sha(p) for p in (Path(__file__),
        REPO/'inspection_agent/paired_port_features.py',ROOT/'artifacts/paired_multimodel_proposal_preregistration_20261003/PLAN.md')}
    OUT.mkdir();records=[];summaries={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        baseline=load(BASE/stage/'report.json');index_path=PREP/stage/'index.json';pins[str(index_path)]=sha(index_path)
        indices=load(index_path)['records'];assert len(indices)==count
        accepted_folder=GEOMETRY/'full_train' if stage=='train' else GEOMETRY/'holdouts'/stage
        totals=dict(current_tp=0,targets=0,old_ceiling=0,extended_ceiling=0,old_candidates=0,extended_candidates=0,extended_new_sources=0)
        for index in indices:
            path=Path(index['path']);assert sha(path)==index['sha256'];pins[str(path)]=index['sha256'];case=load(path);name=index['image']
            entry=next(r for r in baseline['cases'] if r['image']==name)
            teacher,baseline_record=read_current_case(stage,entry,pins);assert teacher==case['teacher']
            assert [case[key]['weight_sha256'] for key in ('teacher','student','feature')]==[TEACHER_SHA,STUDENT_SHA,WEIGHT_SHA]
            fixed=accepted_folder/(Path(name).stem+'_predictions.json');pins[str(fixed)]=sha(fixed);current=load(fixed)['trial'];old=current['all_predictions']
            remaining=5-(len(old)-len(current['primary']));assert remaining>=0
            def native(models):
                pool=proposals(teacher,models) if remaining else []
                return [row for row in pool if not any(box_iou(row['box_xyxy'],prior['box_xyxy'])>=.5 for prior in old)]
            old_pool=native([teacher,case['student']]);extended=native([teacher,case['student'],case['feature'],baseline_record['alternative']])
            save(OUT/(stage+'_'+Path(name).stem+'_proposals.json'),dict(image=name,current=current,old_candidates=old_pool,
                extended_candidates=extended,remaining=remaining,oracle_not_used_for_candidates=True))
            targets=read_targets(stage,name,teacher['predictions']['source_shape'],entry['label_sha256'],pins);m=metric(old,targets)
            assert len(maximum_matching(old,targets))==m['tp']
            old_upper=min(len(maximum_matching(old+old_pool,targets)),m['tp']+remaining)
            extended_upper=min(len(maximum_matching(old+extended,targets)),m['tp']+remaining)
            row=dict(stage=stage,image=name,current_tp=m['tp'],targets=len(targets),old_ceiling=old_upper,extended_ceiling=extended_upper,
                old_candidates=len(old_pool),extended_candidates=len(extended),extended_new_sources=int(extended_upper>old_upper))
            records.append(row)
            for key in totals:totals[key]+=row[key]
        assert totals['current_tp']==load(accepted_folder/'report.json')['summary']['trial']['tp']
        summaries[stage]=dict(totals,old_potential_gain=totals['old_ceiling']-totals['current_tp'],
            extended_potential_gain=totals['extended_ceiling']-totals['current_tp'],not_achieved_accuracy=True)
    assert {p:sha(Path(p)) for p in pins}==pins and paired_runtime_fingerprint(REPO)==frozen
    save(OUT/'report.json',dict(status='complete',summary=summaries,cases=records,pins=pins,
        raw_extension_can_reorder_proposals=True,old_current_prefix_preserved=True,oracle_not_runtime=True,
        model_not_trained_or_inferred=True,no_deployment=True,field_accuracy=False))
    print(str(summaries),flush=True)

if __name__=='__main__':main()
