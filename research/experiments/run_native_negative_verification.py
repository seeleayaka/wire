"""Whole-population precision trial; strong OTHER only, no source tuning."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,cached_alignments,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_RELATIVE,HEAD_SHA
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,paired_features,embeddings
from paired_port_semantics import CachedReferenceSIFT
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004'
HEADS=ROOT/'artifacts/paired_pose_native_training_20261004/heads_oof'
OUT=ROOT/'artifacts/native_negative_verification_20261004'
PLAN=ROOT/'artifacts/native_negative_verification_preregistration_20261004/PLAN.md'

def main():
    if OUT.exists():raise FileExistsError('Preserve negative verification trial')
    import torch
    import cv2
    import dino_feature_diff as dino
    import assembly_auto_review_robust_v3 as registration
    torch.set_num_threads(2);cv2.setNumThreads(1)
    frozen=native_pose_runtime_fingerprint(REPO);groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';groups=load(groupspath)
    source_folds={name:i%3 for i,name in enumerate(sorted(groups['train_sources']))};assert len(source_folds)==192
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,groupspath,referencepath,REPO/HEAD_RELATIVE,
        REPO/'inspection_agent/paired_port_features.py',REPO/'prototype/assembly_auto_review_robust_v3.py')}
    assert pins[str(REPO/HEAD_RELATIVE)]==HEAD_SHA
    old_training=load(ROOT/'artifacts/paired_pose_native_training_20261004/report.json')
    def head(path,full=False):
        digest=sha(path);pins[str(path)]=digest
        if not full:assert old_training['pins'][str(path)]==digest
        weights=torch.load(path,map_location='cpu',weights_only=True);m=torch.nn.Linear(6144,3)
        m.load_state_dict(weights['state_dict'] if full else weights);m.eval().requires_grad_(False);return m
    folds={i:head(HEADS/('fold'+str(i)+'_head.pt')) for i in range(3)};full=head(REPO/HEAD_RELATIVE,True)
    OUT.mkdir();started=time.monotonic();encoder=None;stages={};train_cache={}
    reference=read_image(referencepath);alignments=cached_alignments()
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,no_training=True,p_other_gate=.98,
        no_positive_class_disagreement_rejection=True,classifier_only_OOF=True,field_accuracy=False))
    def finish(failed):
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='rejected' if failed else 'source_precision_pass_requires_actual_gates',failed_stage=failed,stages=stages,pins=pins,
            runtime=frozen,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],failed_stage=failed,seconds=result['seconds']))
    try:
        with CachedReferenceSIFT(reference):
            for stage in ('source_train_oof','full_train','inner','outer'):
                split='train' if stage in ('source_train_oof','full_train') else stage
                folder=OUT/stage;folder.mkdir();rows=[];removed=0
                for i,entry in enumerate(load(BASE/split/'report.json')['cases']):
                    name=entry['image'];stem=Path(name).stem;casepath=SOURCE/('full_train' if split=='train' else split)/(stem+'_predictions.json');pins[str(casepath)]=sha(casepath)
                    current=load(casepath)['trial']['all_predictions'];scores=[];valid=[];alignment=None
                    save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),stage=stage,image=name,completed=i))
                    if stage=='full_train':values,valid,alignment=train_cache[name]
                    elif current:
                        source=DATA/'images'/('val01' if split=='outer' else 'train01')/name;pins[str(source)]=sha(source);image=read_image(source)
                        if name in alignments:
                            provenance,cached=alignments[name];pins[str(provenance)]=sha(provenance)
                            assert cached['image_fingerprints']['source_sha256']==pins[str(source)];alignment=cached['alignment']
                        else:cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2]);boxes=[p['box_xyxy'] for p in current];valid=valid_boxes(boxes,mask)
                            if valid:
                                if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                                selected=[boxes[j] for j in valid];values=paired_features(embeddings(encoder,image,selected),embeddings(encoder,expected,selected))
                            else:values=torch.empty((0,6144))
                        else:values=torch.empty((0,6144))
                    else:values=torch.empty((0,6144))
                    if stage=='source_train_oof':train_cache[name]=(values,valid,alignment)
                    model=folds[source_folds[name]] if stage=='source_train_oof' else full
                    with torch.inference_mode():scores=model(values).softmax(dim=1).tolist() if len(valid) else []
                    assert len(valid)==len(scores)
                    reject={index for index,prob in zip(valid,scores) if prob[0]>=.98}
                    trial=[p for index,p in enumerate(current) if index not in reject];removed+=len(reject)
                    targets=read_targets(split,name,[2736,3648],entry['label_sha256'],pins);oldhits=matches(current,targets)[0];newhits=matches(trial,targets)[0]
                    rows.append(dict(image=name,current=metric(current,targets),trial=metric(trial,targets),lost=sorted(oldhits-newhits),gained=sorted(newhits-oldhits),removed_indices=sorted(reject)))
                    save(folder/(stem+'_predictions.json'),dict(image=name,current=current,trial=trial,valid_context_indices=valid,probabilities=scores,removed_indices=sorted(reject),alignment=alignment,experimental_only=True))
                totals={version:{k:sum(row[version][k] for row in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
                normal=sum(row['trial']['predictions'] for row in rows if row['image'].startswith('normal_'))
                error_gain=totals['trial']['unmatched']<totals['current']['unmatched'] if split=='train' else totals['trial']['unmatched']<=totals['current']['unmatched']
                qualifies=error_gain and not any(row['lost'] for row in rows) and normal==0
                stages[stage]=dict(qualifies=qualifies,summary=totals,removed_cues=removed,normal_cues=normal)
                save(folder/'report.json',dict(status='complete',**stages[stage],cases=rows));print(dict(stage=stage,**stages[stage]),flush=True)
                if not qualifies:finish(stage);return
        finish(None)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':main()
