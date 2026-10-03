"""ALL192 cached original-geometry semantic gates with consensus localization rank."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA
from audit_port_multiscale_acceptance import metric,matches,overlap
from consensus_rank import select
SOURCE=ROOT/'artifacts/fine_native_consensus_20261004/train'
VIEWS=ROOT/'artifacts/paired_fine_tile_views_20261003/full/train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
OUT=ROOT/'artifacts/fine_consensus_rank_20261004'
PLAN=ROOT/'artifacts/fine_consensus_rank_preregistration_20261004/PLAN.md'


def main():
    if OUT.exists():raise FileExistsError('Preserve fixed consensus ranking')
    frozen=native_pose_runtime_fingerprint(REPO)
    oldpath=ROOT/'artifacts/fine_native_consensus_20261004/report.json';old=load(oldpath)
    assert old['status']=='rejected' and old['runtime']==frozen
    assert all(sha(Path(p))==v for p,v in old['pins'].items())
    auditpath=ROOT/'artifacts/fine_native_consensus_replay_20261004/report.json'
    # The upstream may use audit rather than replay folder; resolve only exact known reports.
    if not auditpath.exists():auditpath=ROOT/'artifacts/fine_native_consensus_audit_20261004/report.json'
    previous=ROOT/'artifacts/fine_novel_geometry_20261004/report.json'
    assert load(previous)['status']=='rejected_no_new_geometry_potential'
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'
    names=sorted(load(groupspath)['train_sources']);assert len(names)==192
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,Path(__file__).with_name('consensus_rank.py'),
        Path(__file__).with_name('novel_box_geometry.py'),Path(__file__).with_name('relative_port_box.py'),oldpath,previous,
        groupspath,PREP,REPO/'inspection_agent/paired_native_pose_features.py',REPO/'inspection_agent/paired_port_features.py')}
    indexed={r['image']:r for r in load(PREP)['records']};entries=load(BASE/'train/report.json')['cases'];assert sorted(r['image'] for r in entries)==names
    OUT.mkdir();started=time.monotonic();rows=[];candidate_count=0;folder=OUT/'train';folder.mkdir()
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,train_sources=names,geometric_rank='median_best_IoU_per_distinct_voter',
        probability_gate=.98,distinct_SHA_votes=3,duplicate_IoMin=.5,budget=[5,5],unchanged_boxes=True,
        cached_original_pixel_scores=True,no_training=True,no_validation_read=True,no_deployment=True,field_accuracy=False))
    try:
        for completed,entry in enumerate(entries):
            name=entry['image'];save(OUT/'progress.json',dict(status='running',pid=os.getpid(),image=name,completed=completed,total=192,seconds=round(time.monotonic()-started,2)))
            path=SOURCE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            candidates=copy.deepcopy(case['proposals']);scores=case['probabilities'];assert len(candidates)==len(scores)
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            if candidates:
                pairpath=Path(indexed[name]['path']);assert sha(pairpath)==indexed[name]['sha256'];pins[str(pairpath)]=sha(pairpath);paired=load(pairpath)
                teacher,baseline=read_current_case('train',entry,pins);assert teacher==paired['teacher']
                viewpath=VIEWS/(Path(name).stem+'_predictions.json');pins[str(viewpath)]=sha(viewpath);fine=load(viewpath)
                models=[teacher,paired['student'],paired['feature'],baseline['alternative'],*fine['new_views']];pool=[]
                for model in models:
                    assert model['source_sha256']==pins[str(source)] and model['predictions']['source_shape']==[2736,3648]
                    for row in model['predictions']['merged_predictions']:
                        l,t,r,b=row['box_xyxy']
                        if row['confidence']>.05 and 16<=l<r<=3632 and 16<=t<b<=2720:pool.append((model['weight_sha256'],row))
                for row in candidates:
                    best={}
                    for digest,modelrow in pool:
                        if modelrow['class_id']!=row['class_id']:continue
                        value=overlap(modelrow['box_xyxy'],row['box_xyxy'])
                        if value>=.5:best[digest]=max(value,best.get(digest,0.))
                    assert sorted(best)==row['semantic_model_vote_sha256']
                    row['localization_voter_best_IoU']=best
            trial=select(case['current'],candidates,scores,HEAD_SHA);candidate_count+=len(candidates)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            a=matches(case['current']['all_predictions'],targets)[0];b=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(case['current']['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(b-a),lost=sorted(a-b)))
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=case['current'],trial=trial,proposals=candidates,probabilities=scores,head_sha256=HEAD_SHA))
        assert candidate_count==969
        totals={version:{k:sum(r[version][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        assert totals['current']['tp']==295 and totals['current']['unmatched']==4
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] for r in rows)
        summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal)
        save(folder/'report.json',dict(status='complete',**summary,cases=rows))
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='source_pass_requires_fresh_fine_holdouts_and_actual_gates' if qualifies else 'rejected',
            stages=dict(train=summary),pins=pins,runtime=frozen,candidates=969,no_training=True,no_validation_read=True,
            old_boxes_preserved=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds']));print(result['status'],summary,result['seconds'],flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
