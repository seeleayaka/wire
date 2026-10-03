"""Source committee evaluation, training is explicitly not OOF performance."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,DATA,PROPOSALS,load,save,sha,read_image,cached_alignments,REFERENCE_SHA
from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
from port_semantic_verifier import embeddings
from paired_semantic_committee import select_committee
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric,matches
sys.path.insert(0,str(REPO/'prototype'))
NEW=ROOT/'artifacts/paired_semantic_committee_20261003'


def main():
    import cv2
    import numpy as np
    import torch
    if NEW.exists():raise FileExistsError('Preserve committee evidence')
    trained=load(OUT/'heads_oof/report.json');prepared=load(OUT/'features_train/report.json')
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==prepared['runtime_fingerprint']
    source=OUT/'features_train';assert sha(source/'features.pt')==prepared['aggregate_feature_sha256']
    assert sha(source/'samples.json')==prepared['samples_sha256']
    pins=dict(trained['head_pins'])
    for p in (Path(__file__),Path(__file__).with_name('paired_semantic_committee.py'),source/'features.pt',source/'samples.json',source/'report.json',
              ROOT/'artifacts/paired_semantic_committee_preregistration_20261003/PLAN.md',DATA/'images/train01/normal_073.JPG'):
        pins[str(p)]=sha(p)
    assert {p:sha(Path(p)) for p in pins}==pins
    torch.set_num_threads(2);cv2.setNumThreads(2);heads=[];head_shas=[]
    for fold in range(3):
        path=OUT/'heads_oof'/f'fold{fold}_head.pt';head=torch.nn.Linear(6144,3)
        head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);head.requires_grad_(False).eval()
        heads.append(head);head_shas.append(sha(path))
    assert len(set(head_shas))==3
    metadata=load(source/'samples.json');features=torch.load(source/'features.pt',map_location='cpu',weights_only=True)['features']
    with torch.inference_mode():train_probabilities=[head(features).softmax(dim=1).tolist() for head in heads]
    source_records={row['image']:row for row in prepared['sources']}
    reference=read_image(DATA/'images/train01/normal_073.JPG');assert sha(DATA/'images/train01/normal_073.JPG')==REFERENCE_SHA
    encoder=None;cache=cached_alignments();NEW.mkdir();started=time.monotonic()
    save(NEW/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,all_three_same_class_probability98_required=True,
        exact_old_source_prefix_shared5plus5=True,no_GT_model_inputs=True,training_committee_not_OOF=True,
        no_retraining_no_threshold_search=True,reference_veto_not_used=True,validation_reused=True,no_deployment=True,field_accuracy=False))
    with CachedReferenceSIFT(reference):
        for stage,count in (('train',192),('inner',48),('outer',30)):
            path=BASE/stage/'report.json';pins[str(path)]=sha(path);entries=load(path)['cases'];assert len(entries)==count
            folder=NEW/stage;folder.mkdir();records=[]
            for index,entry in enumerate(entries):
                name=entry['image'];save(NEW/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=count,seconds=round(time.monotonic()-started,2)))
                path=PROPOSALS/(stage+'_'+Path(name).stem+'_proposals.json');pins[str(path)]=sha(path);candidate=load(path);current=candidate['current']
                probabilities=[[],[],[]];native=[];alignment=None;status='no_candidate_exact_short_circuit'
                if stage=='train':
                    indices=[i for i,row in enumerate(metadata) if row['image']==name and row['kind']=='novel_weak_proposal']
                    native=[metadata[i]['proposal'] for i in indices];probabilities=[[member[i] for i in indices] for member in train_probabilities]
                    alignment=source_records[name]['alignment'];status='cached_train_features_committee_not_OOF'
                elif candidate['candidates']:
                    path=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(path)]=sha(path);image=read_image(path)
                    cached=cache.get(name)
                    if cached:
                        assert cached[1]['image_fingerprints']['source_sha256']==pins[str(path)]
                        pins[str(cached[0])]=sha(cached[0]);alignment=cached[1]['alignment']
                    else:
                        import assembly_auto_review_robust_v3 as registration
                        cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                    if alignment['alignment_quality']['reliable']:
                        expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                        all_native=candidate['candidates'];native=[all_native[i] for i in valid_boxes([row['box_xyxy'] for row in all_native],valid)]
                        if encoder is None:
                            import dino_feature_diff as dino
                            encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
                        boxes=[row['box_xyxy'] for row in native]
                        with torch.inference_mode():
                            vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                            probabilities=[head(vectors).softmax(dim=1).tolist() for head in heads]
                        status='fresh_source_paired_features' if boxes else 'abstained_invalid_context'
                    else:status='abstained_registration'
                trial=select_committee(current,native,probabilities,head_shas)
                save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,
                    proposals=native,member_probabilities=probabilities,alignment=alignment,feature_status=status))
                targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
                records.append(dict(image=name,current=metric(old,targets),trial=metric(new,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                    additions=len(trial['paired_committee_additions']),feature_status=status))
                save(folder/'partial.json',dict(cases=records))
            totals={v:{k:sum(row[v][k] for row in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(row['trial']['predictions'] for row in records if row['image'].startswith('normal_'))
            gain=totals['trial']['tp']>=totals['current']['tp'] if stage=='outer' else totals['trial']['tp']>totals['current']['tp']
            qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(row['lost'] for row in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
                training_committee_not_OOF=stage=='train',validation_reused=True,real_current_ROI_Qt_SAM_pending=True,no_deployment=True,field_accuracy=False))
            print(str(dict(stage=stage,qualifies=qualifies,summary=totals)),flush=True)
            if not qualifies:save(NEW/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
    save(NEW/'progress.json',dict(status='committee_source_pass_requires_real_ROI_Qt_SAM_acceptance',no_deployment=True))


if __name__=='__main__':main()
