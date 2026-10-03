"""Independent weak-label scoring of actual displayed cues, partial or complete."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image
from current_port_baseline_audit import BASE
from actual_port_native_rows import actual_native_rows
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
SOURCE=ROOT/'artifacts/native_exposure_stress_20261004'


def independent_score(predictions,targets):
    overlaps=[]
    for pi,p in enumerate(predictions):
        for ti,t in enumerate(targets):
            if p['class_id']!=t['class_id']:continue
            a,b=p['box_xyxy'],t['box']
            width=max(0,min(a[2],b[2])-max(a[0],b[0]))
            height=max(0,min(a[3],b[3])-max(a[1],b[1]))
            intersection=width*height
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            overlaps.append((intersection/union if union>0 else 0.,pi,ti))
    used_predictions,used_targets=set(),set()
    for value,pi,ti in sorted(overlaps,reverse=True):
        if value<.5:continue
        if pi not in used_predictions and ti not in used_targets:
            used_predictions.add(pi);used_targets.add(ti)
    return dict(tp=len(used_targets),unmatched=len(predictions)-len(used_predictions),
        fn=len(targets)-len(used_targets),predictions=len(predictions),targets=len(targets))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    partial=parser.parse_args().partial
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    protocol=load(SOURCE/'protocol.json')
    path=SOURCE/('partial.json' if partial else 'report.json')
    report=load(path);cases=report['cases']
    assert cases and (partial or report['status']=='complete')
    assert partial or len(cases)==78
    out=ROOT/'artifacts'/('native_exposure_actual_replay_20261004_'+(str(len(cases))+'_partial' if partial else 'complete'))
    if out.exists():raise FileExistsError('Preserve actual exposure replay')
    frozen=native_pose_runtime_fingerprint(REPO)
    assert frozen==protocol['runtime']
    assert all(sha(Path(p))==v for p,v in protocol['pins'].items())
    expected=[(stage,row) for stage in ('inner','outer') for row in load(BASE/stage/'report.json')['cases']]
    assert [(r['stage'],r['image']) for r in cases]==[(s,r['image']) for s,r in expected[:len(cases)]]
    pins={};summary={}
    for (stage,entry),row in zip(expected,cases):
        source=DATA/'images'/('val01' if stage=='outer' else 'train01')/row['image']
        assert sha(source)==row['original_sha256']
        reportpath=Path(row['report']);resultpath=Path(row['result'])
        medianpath=resultpath.with_name('accepted_median_ports.json')
        initial=load(reportpath);native=load(resultpath);median=load(medianpath)
        augmented=Path(initial['inspection'])
        assert sha(augmented)==row['augmented_sha256']==initial['image_fingerprints']['source_sha256']
        original=read_image(source)
        dark=np.clip(np.rint(original.astype(np.float32)*.85),0,255).astype(np.uint8)
        assert np.array_equal(read_image(augmented),dark),row['image']
        for p in (source,reportpath,resultpath,medianpath,augmented):pins[str(p)]=sha(p)
        for key,value in median.items():
            if key=='supplementary_hints':assert native[key][:len(value)]==value
            else:assert native[key]==value
        assert len(native.get('rescue_hints',[]))<=5 and len(native.get('supplementary_hints',[]))<=5
        old=actual_native_rows(initial,median);new=actual_native_rows(initial,native)
        assert new[:len(old)]==old and len(new)-len(old)==row['added']
        label=DATA/'labels'/('val01' if stage=='outer' else 'train01')/(Path(row['image']).stem+'.txt')
        assert sha(label)==entry['label_sha256'];pins[str(label)]=sha(label)
        h,w=original.shape[:2];targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,
                box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        for version,predictions in (('median',old),('native',new)):
            scores=independent_score(predictions,targets)
            assert scores==row[version],(row['image'],version)
            totals=summary.setdefault(stage,{}).setdefault(version,{k:0 for k in scores})
            for k,value in scores.items():totals[k]+=value
        assert row['safety_abstention']==(median['status']!='applied')
        assert row['native_reason']==native.get('native_pose_policy',{}).get('fallback_reason')
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
    if not partial:
        assert summary==report['summary']
        assert all(sha(Path(p))==v for p,v in report['pins'].items())
    out.mkdir()
    result=dict(status='pass',cases=len(cases),partial_only=partial,summary=summary,
        source_report_sha256=sha(path) if not partial else None,
        actual_hint_forward_link_verified=True,independent_IoU_and_GT_scoring=True,
        all_lossless_exposure_pixels_verified=True,no_deployment=True,field_accuracy=False,
        auditor_sha256=sha(Path(__file__)))
    save(out/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
