"""Complete-source-only post-selection miss funnel. No runtime GT policy.

Count checkpoint support once per actual SHA; many views are not new voters.
The report diagnoses which generic stage remains weak, not threshold selection,
achievable gain or cross-cabinet electrical fault accuracy.
"""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from audit_allport480_source import score,iou


def miss_reason(case,target,roles):
    cls=target['class_id'];box=target['box'];best={digest:0. for digest in roles.values()}
    for view in case['new_voter_views']:
        digest=view['weight_sha256']
        if digest not in best:raise ValueError('unregistered diagnosis voter')
        for row in view['predictions']['merged_predictions']:
            if row['class_id']==cls and row['confidence']>.05:
                best[digest]=max(best[digest],iou(row['box_xyxy'],box))
    unique=sum(value>=.5 for value in best.values())
    candidates=[(index,row,iou(row['box_xyxy'],box)) for index,row in enumerate(case['proposals']) if row['class_id']==cls]
    localized=[(index,row,value) for index,row,value in candidates if value>=.5]
    high=[(index,row,value) for index,row,value in localized
          if case['probabilities'][index][cls+1]>=.98 and max(range(3),key=lambda k:case['probabilities'][index][k])==cls+1]
    if not case['eligible']:
        reason='policy_'+case['skip_reason']
    elif localized:
        # Actual audited proposals already passed their own three-voter gate.
        # Raw boxes can cover the pose without directly covering GT at IoU.5;
        # do not mistake that indirect support for a missing detector vote.
        reason=('localized_high_semantics_but_novelty_rank_or_budget_blocks' if high
                else 'frozen_paired_semantics_blocks_localized_pose')
    elif unique<3:
        reason='fewer_than_three_actual_checkpoint_supports'
    else:
        reason='no_valid_class_IoU50_pose_after_geometry_context'
    return dict(reason=reason,raw_voter_best_IoU=best,unique_checkpoint_IoU50_supports=unique,
        best_valid_pose_IoU=max((value for _,_,value in candidates),default=0.),
        localized_valid_poses=len(localized),localized_high_semantic_poses=len(high),
        highest_localized_target_probability=max((case['probabilities'][index][cls+1] for index,_,_ in localized),default=None))


def main(source,audit,out):
    source=Path(source).resolve();audit=Path(audit).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('preserve full miss funnel')
    report=load(source/'report.json');independent=load(audit);protocol=load(source/'protocol.json')
    if (len(report['cases'])!=192 or independent['status']!='pass' or independent['source_report_sha256']!=sha(source/'report.json')
        or len(independent['source_case_sha256'])!=192):raise ValueError('requires final independently replayed source192')
    if any(sha(p)!=d for p,d in report['pins'].items()) or native_pose_runtime_fingerprint(REPO)!=report['runtime']:
        raise ValueError('source input/runtime drift')
    pins={str(p):sha(p) for p in (Path(__file__),source/'report.json',source/'protocol.json',audit)}
    reasons={};rows=[];gains=[];burden=[]
    for item in report['cases']:
        name=item['image'];path=source/'train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path)
        if pins[str(path)]!=independent['source_case_sha256'][str(path)]:raise ValueError('audited source case changed')
        case=load(path);label=DATA/'labels/train01'/(Path(name).stem+'.txt');pins[str(label)]=sha(label)
        if pins[str(label)]!=report['pins'][str(label)]:raise ValueError('original source labels changed')
        targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736]))
        current,current_hits=score(case['current']['all_predictions'],targets)
        trial,trial_hits=score(case['trial']['all_predictions'],targets)
        if current!=item['current'] or trial!=item['trial']:raise ValueError('post-selection metrics drift')
        for target_index in sorted(set(range(len(targets)))-trial_hits):
            target=targets[target_index];detail=miss_reason(case,target,protocol['roles'])
            reasons[detail['reason']]=reasons.get(detail['reason'],0)+1
            rows.append(dict(image=name,target_index=target_index,**target,**detail))
        for index in sorted(trial_hits-current_hits):gains.append(dict(image=name,target_index=index,**targets[index]))
        if trial['unmatched']>current['unmatched']:
            burden.append(dict(image=name,added_unmatched=trial['unmatched']-current['unmatched'],dataset_normal_control=name.startswith('normal_')))
    if len(rows)!=report['summary']['trial']['fn'] or len(gains)!=report['summary']['trial']['tp']-report['summary']['current']['tp']:
        raise ValueError('miss/gain totals differ from completed source result')
    if any(sha(p)!=d for p,d in pins.items()) or any(sha(p)!=d for p,d in report['pins'].items()):raise ValueError('funnel inputs changed')
    out.mkdir();save(out/'report.json',dict(status='complete_post_selection_diagnosis',source_report_sha256=pins[str(source/'report.json')],
        sources=192,roles=protocol['roles'],summary=report['summary'],miss_taxonomy=reasons,remaining_misses=rows,
        new_source_matches=gains,new_unmatched_burden=burden,pins=pins,source_candidate_qualifies=report['qualifies'],
        no_model_inference=True,no_GT_policy_inputs=True,no_heldout_reads=True,no_threshold_change=True,
        same_weight_views_one_support=True,no_deployment=True,field_accuracy=None,
        warning='Repeated TRAIN192 diagnosis only; match counts are not new cabinet field/electrical-connection accuracy.'))
    print(dict(status='complete_post_selection_diagnosis',miss_taxonomy=reasons,new_source_matches=len(gains),new_unmatched_burden=burden),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'artifacts/allport480_source_20261005')
    parser.add_argument('--audit',type=Path,default=ROOT/'artifacts/allport480_source_audit_20261005/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/allport480_miss_funnel_20261005');args=parser.parse_args()
    main(args.source,args.audit,args.output)
