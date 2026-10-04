"""Label-free ALL192 missing-voter readiness; no network-model inference."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source,valid_boxes
from unresolved_voter_seeds import seeds
SOURCE=ROOT/'artifacts/fine_consensus_rank_20261004'
FINE=ROOT/'artifacts/paired_fine_tile_views_20261003/full/train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
ALIGN=ROOT/'artifacts/paired_port_semantics_20261003/features_train'
OUT=ROOT/'artifacts/two_vote_resolution_preparation_20261004'
PLAN=ROOT/'artifacts/two_vote_resolution_preregistration_20261004/PLAN.md'


def main():
    import cv2
    cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve missing-checkpoint preparation')
    previous=load(SOURCE/'report.json');assert previous['status']=='source_pass_requires_fresh_fine_holdouts_and_actual_gates'
    auditpath=ROOT/'artifacts/fine_consensus_rank_replay_20261004/report.json';audit=load(auditpath)
    assert audit['status']=='pass' and audit['source_report_sha256']==sha(SOURCE/'report.json')
    assert all(sha(Path(p))==v for p,v in previous['pins'].items())
    frozen=native_pose_runtime_fingerprint(REPO);assert previous['runtime']==frozen
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';names=sorted(load(groupspath)['train_sources']);assert len(names)==192
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,Path(__file__).with_name('unresolved_voter_seeds.py'),
        Path(__file__).with_name('novel_box_geometry.py'),Path(__file__).with_name('relative_port_box.py'),
        PREP,groupspath,SOURCE/'report.json',auditpath,referencepath,REPO/'inspection_agent/paired_native_pose_features.py',REPO/'inspection_agent/paired_port_features.py')}
    indexed={r['image']:r for r in load(PREP)['records']};reference=read_image(referencepath);entries=load(BASE/'train/report.json')['cases']
    assert sorted(r['image'] for r in entries)==names
    OUT.mkdir();started=time.monotonic();folder=OUT/'train';folder.mkdir();rows=[]
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,train_sources=names,no_GT_read=True,no_detector_or_DINO_inference=True,
        research_prefix_298_not_installed=True,final_cues_require_three_checkpoint_votes=True,no_deployment=True,field_accuracy=False))
    for completed,entry in enumerate(entries):
        name=entry['image'];save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=completed,total=192,image=name,seconds=round(time.monotonic()-started,2)))
        sourcepath=DATA/'images/train01'/name;source_sha=sha(sourcepath);pins[str(sourcepath)]=source_sha
        currentpath=SOURCE/'train'/(Path(name).stem+'_predictions.json');pins[str(currentpath)]=sha(currentpath);current=load(currentpath)['trial']
        pairpath=Path(indexed[name]['path']);assert sha(pairpath)==indexed[name]['sha256'];pins[str(pairpath)]=sha(pairpath);pair=load(pairpath)
        teacher,old=read_current_case('train',entry,pins);assert teacher==pair['teacher']
        finepath=FINE/(Path(name).stem+'_predictions.json');pins[str(finepath)]=sha(finepath);fine=load(finepath)
        models=[teacher,pair['student'],pair['feature'],old['alternative'],*fine['new_views']]
        weightset=set(m['weight_sha256'] for m in models);assert len(weightset)==3
        remaining=5-(len(current['all_predictions'])-len(current['primary']));assert remaining>=0
        native=seeds(models,current,source_sha,(2736,3648)) if remaining else []
        alignment=None;before=len(native);reason=None
        if native:
            path=ALIGN/(Path(name).stem+'_source.json');pins[str(path)]=sha(path);aligned=load(path);assert aligned['source_sha256']==source_sha
            alignment=aligned['alignment']
            if alignment.get('alignment_quality',{}).get('reliable'):
                _,mask=expected_in_source(reference,alignment['source_to_reference_homography'],(2736,3648))
                native=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
            else:native=[];reason='original_registration_abstention'
        if not native and reason is None:reason='no_remaining_budget_or_valid_exact_two_vote_seed'
        path=folder/(Path(name).stem+'_seeds.json')
        save(path,dict(image=name,current=current,models=models,seeds=native,alignment=alignment,source_sha256=source_sha,
            missing_weight_by_seed=[sorted(weightset-set(p['semantic_model_vote_sha256'])) for p in native],remaining_budget=remaining,skip_reason=reason))
        assert all(len(weightset-set(p['semantic_model_vote_sha256']))==1 for p in native)
        rows.append(dict(image=name,remaining_budget=remaining,raw_seeds=before,valid_seeds=len(native),unique_parents=len(set(p['pose_parent_seed_id'] for p in native)),
            path=str(path),sha256=sha(path),source_sha256=source_sha,skip_reason=reason))
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
    result=dict(status='complete',sources=192,valid_seeds=sum(r['valid_seeds'] for r in rows),sources_with_seeds=sum(r['valid_seeds']>0 for r in rows),
        cases=rows,pins=pins,runtime=frozen,no_GT_read=True,no_detector_or_DINO_inference=True,
        final_cues_require_three_checkpoint_votes=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
    save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='complete',seconds=result['seconds']));print({k:v for k,v in result.items() if k not in ('cases','pins','runtime')},flush=True)


if __name__=='__main__':main()
