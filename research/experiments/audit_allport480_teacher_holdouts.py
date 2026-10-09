"""Independent full-stage view/selector/GT replay; no inference or fitting."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA
from replacement_voters import check_prefix
from consensus_rank import select
from audit_allport480_source import audit_views,audit_probabilities,score,iou
from run_allport480_teacher_holdouts import stage_qualifies,check_stage_membership,GROUPS,CURRENT

SOURCE=ROOT/'artifacts/allport480_teacher_holdouts_20261005'
OUT=ROOT/'artifacts/allport480_teacher_holdout_audit_20261005'


def main():
    if OUT.exists():raise FileExistsError('preserve independent full development audit')
    digest=sha(SOURCE/'report.json');protocol_digest=sha(SOURCE/'protocol.json')
    report=load(SOURCE/'report.json');protocol=load(SOURCE/'protocol.json')
    if report['status'] not in ('rejected','development_pass_requires_independent_actual_reference_ROI_Qt_SAM'):
        raise ValueError('final validation result required, not live partial progress')
    if any(sha(p)!=d for p,d in report['pins'].items()) or native_pose_runtime_fingerprint(REPO)!=report['runtime']:
        raise ValueError('frozen validation input/runtime drift')
    roles=protocol['roles'];allowed=set(roles.values());case_pins={};stages={};view_count=0;proposal_count=0
    if len(allowed)!=3 or protocol['fresh_semantic_scores_only'] is not True:raise ValueError('fixed fresh three-role experiment required')
    order=list(report['stages'])
    groups=load(GROUPS)
    if report['pins'].get(str(GROUPS))!=sha(GROUPS):raise ValueError('fixed original groups are not bound')
    if order not in (['inner'],['inner','outer']):raise ValueError('wrong stage population/order')
    for stage in order:
        if stage=='outer' and not stages['inner']['qualifies']:raise ValueError('OUTER read after failed INNER')
        folder=SOURCE/stage;stagepath=folder/'report.json';case_pins[str(stagepath)]=sha(stagepath);result=load(stagepath)
        count,base_tp,base_fp=(48,68,0) if stage=='inner' else (30,40,1)
        rows=result['cases'];names=[r['image'] for r in rows]
        if result['status']!='complete' or len(rows)!=count or len(set(names))!=count:raise ValueError('full fixed stage required')
        acceptedstagepath=CURRENT/stage/'report.json'
        if report['pins'].get(str(acceptedstagepath))!=sha(acceptedstagepath):raise ValueError('accepted membership is not bound')
        check_stage_membership(stage,names,[r['image'] for r in load(acceptedstagepath)['cases']],groups['train_sources'],groups['inner_val_sources'])
        if {p.name for p in folder.glob('*_predictions.json')}!={Path(n).stem+'_predictions.json' for n in names}:
            raise ValueError('validation output inventory mismatch')
        replay=[];abstentions={}
        for item in rows:
            name=item['image'];path=folder/(Path(name).stem+'_predictions.json');case_pins[str(path)]=sha(path);case=load(path)
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
            if case['image']!=name or case['roles']!=roles or case['head_sha256']!=HEAD_SHA or sha(source)!=case['source_sha256']:
                raise ValueError('bound validation source/role/head changed')
            if report['pins'][str(source)]!=case['source_sha256']:raise ValueError('source is not pinned in completed result')
            check_prefix(case['original'],case['current'],case['trial'])
            acceptedpath=CURRENT/stage/(Path(name).stem+'_predictions.json')
            if report['pins'].get(str(acceptedpath))!=sha(acceptedpath) or case['current']!=load(acceptedpath)['trial']:
                raise ValueError('current prefix is not the accepted validation output')
            if case['eligible']:
                if case['alignment']['alignment_quality']['reliable'] is not True:raise ValueError('unreliable reference acquired views')
                view_count+=audit_views(case['new_voter_views'],roles,case['source_sha256'],[2736,3648])
                pool=[(v['weight_sha256'],row) for v in case['new_voter_views'] for row in v['predictions']['merged_predictions']
                    if row['confidence']>.05 and 16<=row['box_xyxy'][0]<row['box_xyxy'][2]<=3632 and 16<=row['box_xyxy'][1]<row['box_xyxy'][3]<=2720]
                for proposal in case['proposals']:
                    best={}
                    for weight,row in pool:
                        if row['class_id']!=proposal['class_id']:continue
                        value=iou(row['box_xyxy'],proposal['box_xyxy'])
                        if value>=.5:best[weight]=max(value,best.get(weight,0))
                    if set(best)!=allowed or sorted(best)!=proposal['semantic_model_vote_sha256'] or best!=proposal['localization_voter_best_IoU']:
                        raise ValueError('new validation cue lacks three actual checkpoint localization votes')
            else:
                if case['new_voter_views'] or case['proposals'] or case['probabilities']:raise ValueError('abstention acquired inference evidence')
                reason=case['skip_reason'];abstentions[reason]=abstentions.get(reason,0)+1
            audit_probabilities(case['proposals'],case['probabilities']);proposal_count+=len(case['proposals'])
            if select(case['current'],case['proposals'],case['probabilities'],HEAD_SHA)!=case['trial']:
                raise ValueError('fresh validation selector replay mismatch')
            label=DATA/'labels'/('val01' if stage=='outer' else 'train01')/(Path(name).stem+'.txt')
            if sha(label)!=report['pins'][str(label)]:raise ValueError('original label drift')
            targets=[]
            for line in label.read_text(encoding='utf-8').splitlines():
                cls,cx,cy,w,h=map(float,line.split())
                if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736]))
            metrics={};hits={}
            for version in ('original','current','trial'):metrics[version],hits[version]=score(case[version]['all_predictions'],targets)
            row=dict(image=name,**metrics,gained=sorted(hits['trial']-hits['current']),lost=sorted(hits['current']-hits['trial']),
                lost_original=sorted(hits['original']-hits['trial']),skip_reason=case['skip_reason'])
            if row!=item:raise ValueError('independent stage metric mismatch')
            replay.append(row)
        totals={v:{k:sum(r[v][k] for r in replay) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('original','current','trial')}
        if totals['current']['tp']!=base_tp or totals['current']['unmatched']!=base_fp:raise ValueError('accepted development baseline changed')
        normal=sum(r['trial']['predictions'] for r in replay if r['image'].startswith('normal_'))
        stages[stage]=dict(qualifies=stage_qualifies(stage,totals,normal,replay),summary=totals,normal_cues=normal,count=count)
        if stages[stage]!=report['stages'][stage] or any(result[k]!=v for k,v in stages[stage].items()):raise ValueError('independent full stage gate differs')
    passed=order==['inner','outer'] and all(r['qualifies'] for r in stages.values())
    if passed!=(report['status']=='development_pass_requires_independent_actual_reference_ROI_Qt_SAM'):raise ValueError('overall validation verdict mismatch')
    if (sha(SOURCE/'report.json')!=digest or sha(SOURCE/'protocol.json')!=protocol_digest or any(sha(p)!=d for p,d in case_pins.items())
        or any(sha(p)!=d for p,d in report['pins'].items()) or native_pose_runtime_fingerprint(REPO)!=report['runtime']):raise ValueError('evidence drift during independent replay')
    OUT.mkdir();save(OUT/'report.json',dict(status='pass',candidate_development_qualifies=passed,stages=stages,source_report_sha256=digest,
        source_protocol_sha256=protocol_digest,source_case_sha256=case_pins,actual_view_replays=view_count,proposal_replays=proposal_count,
        auditor_sha256=sha(__file__),independent_original_labels_and_matching=True,no_inference=True,no_fitting=True,no_deployment=True,
        field_accuracy=None,actual_reference_ROI_Qt_SAM_still_required=True,reused_development_groups=True,
        warning='Repeated development image groups, not independent field accuracy or electrical continuity recognition.'))
    print(dict(status='pass',candidate_development_qualifies=passed,stages=stages),flush=True)


if __name__=='__main__':main()
