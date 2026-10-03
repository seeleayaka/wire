"""Matched actual original/exposure endpoint scoring, partial INNER or full78."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image
from current_port_baseline_audit import BASE
from actual_port_native_rows import actual_native_rows
from verify_native_exposure_actual_scores import independent_score
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
SOURCE=ROOT/'artifacts/native_original_pairing_20261004'
EXPOSURE=ROOT/'artifacts/native_exposure_stress_20261004'


def hits(predictions,targets):
    pairs=[]
    for pi,p in enumerate(predictions):
        for ti,t in enumerate(targets):
            if p['class_id']!=t['class_id']:continue
            a,b=p['box_xyxy'],t['box'];area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
            pairs.append((area/union if union>0 else 0,pi,ti))
    usedp=set();usedt=set()
    for score,pi,ti in sorted(pairs,reverse=True):
        if score>=.5 and pi not in usedp and ti not in usedt:usedp.add(pi);usedt.add(ti)
    return usedt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial-inner',action='store_true');partial=parser.parse_args().partial_inner
    out=ROOT/'artifacts'/('native_original_actual_replay_20261004_inner48_partial' if partial else 'native_original_actual_replay_20261004_complete')
    if out.exists():raise FileExistsError('Preserve matched actual score audit')
    protocol=load(SOURCE/'protocol.json');frozen=native_pose_runtime_fingerprint(REPO);assert frozen==protocol['runtime']
    assert all(sha(Path(p))==v for p,v in protocol['pins'].items())
    path=SOURCE/('partial.json' if partial else 'report.json');report=load(path);snapshot_sha=sha(path)
    cases=[r for r in report['cases'] if r['stage']=='inner'] if partial else report['cases']
    assert len(cases)==(48 if partial else 78)
    if not partial:assert report['status']=='complete'
    expected=[(s,r) for s in (('inner',) if partial else ('inner','outer')) for r in load(BASE/s/'report.json')['cases']]
    assert [(r['stage'],r['image']) for r in cases]==[(s,r['image']) for s,r in expected]
    dark_report=load(EXPOSURE/'report.json');assert dark_report['status']=='complete'
    dark={(r['stage'],r['image']):r for r in dark_report['cases']};summary={};pins={};lost=0;gained=0;abstentions={}
    out.mkdir();save(out/'case_snapshot.json',dict(cases=cases,read_at_sha256=snapshot_sha,partial=partial))
    for (stage,entry),row in zip(expected,cases):
        source=DATA/'images'/('val01' if stage=='outer' else 'train01')/row['image'];assert sha(source)==row['source_sha256']
        reportpath=Path(row['report']);resultpath=Path(row['result']);medianpath=resultpath.with_name('accepted_median_ports.json')
        original_report=load(reportpath);native=load(resultpath);median=load(medianpath)
        assert original_report['image_fingerprints']['source_sha256']==row['source_sha256'] and Path(original_report['inspection'])==source
        for key,value in median.items():
            if key=='supplementary_hints':assert native[key][:len(value)]==value
            else:assert native[key]==value
        assert len(native.get('rescue_hints',[]))<=5 and len(native.get('supplementary_hints',[]))<=5
        darkcase=dark[(stage,row['image'])];dark_initial=load(Path(darkcase['report']));dark_result=load(Path(darkcase['result']))
        original_rows=actual_native_rows(original_report,native);exposure_rows=actual_native_rows(dark_initial,dark_result)
        old_rows=actual_native_rows(original_report,median);assert original_rows[:len(old_rows)]==old_rows
        label=DATA/'labels'/('val01' if stage=='outer' else 'train01')/(Path(row['image']).stem+'.txt');assert sha(label)==entry['label_sha256']
        h,w=read_image(source).shape[:2];targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        for version,values in (('original',original_rows),('exposure',exposure_rows)):
            score=independent_score(values,targets);assert score==row[version]
            totals=summary.setdefault(stage,{}).setdefault(version,{k:0 for k in score})
            for k in score:totals[k]+=score[k]
        a=hits(original_rows,targets);b=hits(exposure_rows,targets)
        assert sorted(a-b)==row['lost_under_exposure'] and sorted(b-a)==row['gained_under_exposure']
        lost+=len(a-b);gained+=len(b-a)
        assert row['original_abstention']==(median['status']!='applied') and row['exposure_abstention']==darkcase['safety_abstention']
        counts=abstentions.setdefault(stage,{'original':0,'exposure':0});counts['original']+=row['original_abstention'];counts['exposure']+=row['exposure_abstention']
        for p in (source,label,reportpath,resultpath,medianpath,Path(darkcase['report']),Path(darkcase['result'])):pins[str(p)]=sha(p)
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
    latest=load(path)
    assert latest['cases'][:len(cases)]==cases # completed prefix immutable while worker may append
    if not partial:assert summary==report['summary'] and lost==report['lost_targets'] and gained==report['gained_targets'] and sha(path)==snapshot_sha
    result=dict(status='pass',sources=len(cases),partial_only=partial,summary=summary,lost_targets=lost,gained_targets=gained,
        abstentions=abstentions,source_read_sha256=snapshot_sha,independent_actual_hint_forward_link_GT_and_paired_target_replay=True,
        pins=pins,source_protocol_sha256=sha(SOURCE/'protocol.json'),auditor_sha256=sha(Path(__file__)),
        no_training=True,no_deployment=True,synthetic_same_scene_not_field_accuracy=True,field_accuracy=False)
    save(out/'report.json',result);print({k:v for k,v in result.items() if k!='pins'},flush=True)


if __name__=='__main__':main()
