"""TRAIN-only cache readiness while the first source run is still active.

No model/GT/heldout reads; no claim that incomplete source data passed. Protect
each completed parent's actual view schema and plan missing old-student views.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from run_allport480_source import FINE,PAIR,RESEARCH,cached_view
from inspection_agent.teacher_student_port_support import STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from audit_allport480_source import audit_views,audit_probabilities

SOURCE=ROOT/'artifacts/allport480_source_20261005'
OUT=ROOT/'artifacts/allport480_teacher_cache_readiness_20261005'


def main():
    if OUT.exists():raise FileExistsError('preserve cache readiness snapshot')
    protocol=load(SOURCE/'protocol.json');names=sorted(protocol['train_sources'])
    fine=load(FINE/'report.json');index={r['image']:r for r in load(PAIR)['records']}
    if len(names)!=192 or len(set(names))!=192:raise ValueError('TRAIN source membership drift')
    frozen=native_pose_runtime_fingerprint(REPO)
    if frozen!=protocol['runtime'] or sha(REPO/STUDENT_RELATIVE)!=STUDENT_SHA:raise ValueError('mainline/model drift')
    pins={str(p):sha(p) for p in (Path(__file__),SOURCE/'protocol.json',PAIR,FINE/'report.json',REPO/STUDENT_RELATIVE)}
    rows=[];counts=dict(eligible=0,abstained=0,actual_old_student_native=0,missing_old_student_native=0,
        actual_old_student_fine=0,missing_old_student_fine=0,completed_parent_sources=0,actual_parent_views_replayed=0)
    # Snapshot the completed inventory once; do not chase moving progress.
    completed={p.name:p for p in (SOURCE/'train').glob('*_predictions.json')}
    for name in names:
        source=DATA/'images/train01'/name;digest=sha(source);pins[str(source)]=digest
        priorpath=RESEARCH/'train'/(Path(name).stem+'_predictions.json');pins[str(priorpath)]=sha(priorpath)
        current=load(priorpath)['trial'];finepath=FINE/'train'/(Path(name).stem+'_predictions.json')
        pins[str(finepath)]=sha(finepath);case=load(finepath);alignment=case['alignment']
        remaining=5-(len(current['all_predictions'])-len(current['primary']))
        eligible=remaining>0 and alignment is not None and alignment.get('alignment_quality',{}).get('reliable') is True
        row=dict(image=name,eligible=eligible)
        if eligible:
            if fine['pins'][str(source)]!=digest:raise ValueError('cached frame changed')
            counts['eligible']+=1;pairpath=Path(index[name]['path']);pins[str(pairpath)]=sha(pairpath)
            if pins[str(pairpath)]!=index[name]['sha256']:raise ValueError('native evidence drift')
            pair=load(pairpath);native=pair['student']
            if 'windows' in native['predictions']:
                cached_view(native,digest,STUDENT_SHA,1280,960);counts['actual_old_student_native']+=1;row['native']='verified_actual'
            else:
                if native['predictions']['merged_predictions'] or native['predictions']['edge_kept_predictions']:
                    raise ValueError('nonempty student cache lacks actual recipe')
                counts['missing_old_student_native']+=1;row['native']='missing_placeholder_requires_fresh'
            views=[v for v in case['new_views'] if v['weight_sha256']==STUDENT_SHA]
            if views:
                if len(views)!=1:raise ValueError('ambiguous old-student fine recipe')
                cached_view(views[0],digest,STUDENT_SHA,960,720);counts['actual_old_student_fine']+=1;row['fine']='verified_actual'
            else:counts['missing_old_student_fine']+=1;row['fine']='missing_requires_fresh'
        else:counts['abstained']+=1
        parent=completed.get(Path(name).stem+'_predictions.json')
        if parent:
            pins[str(parent)]=sha(parent);saved=load(parent)
            if saved['eligible']!=eligible or saved['source_sha256']!=digest:raise ValueError('parent source/frame eligibility changed')
            audit_probabilities(saved['proposals'],saved['probabilities'])
            if eligible:counts['actual_parent_views_replayed']+=audit_views(saved['new_voter_views'],protocol['roles'],digest,[2736,3648])
            elif saved['new_voter_views'] or saved['proposals']:raise ValueError('abstention gained evidence')
            counts['completed_parent_sources']+=1
        rows.append(row)
    if any(sha(p)!=d for p,d in pins.items()) or native_pose_runtime_fingerprint(REPO)!=frozen:raise ValueError('cache readiness evidence drift')
    OUT.mkdir();save(OUT/'report.json',dict(status='complete_readiness_snapshot_not_source_acceptance',counts=counts,cases=rows,pins=pins,
        runtime=frozen,parent_source_run_status_observed=load(SOURCE/'progress.json')['status'],no_GT_reads=True,
        no_heldout_reads=True,no_model_inference=True,no_deployment=True,field_accuracy=None,
        teacher_source_not_launched=True,warning='Completed parent outputs only; this is NOT the final192 source audit or acceptance.'))
    print(dict(status='complete_readiness_snapshot_not_source_acceptance',counts=counts),flush=True)


if __name__=='__main__':main()
