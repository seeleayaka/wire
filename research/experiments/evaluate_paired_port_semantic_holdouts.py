"""Fresh GT-free conditional candidate inference before inner/outer scoring."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,DATA,PROPOSALS,load,save,sha,read_image,cached_alignments,REFERENCE_SHA
from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
from port_semantic_verifier import embeddings
from paired_port_semantic_selection import select
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric,matches


def main():
    destination=OUT/'holdouts'
    if destination.exists():raise FileExistsError('Preserve holdout evidence')
    trained=load(OUT/'head_full/report.json');assert trained['status']=='complete' and trained['no_validation_training']
    weight=Path(trained['checkpoint']['path']);assert sha(weight)==trained['checkpoint']['sha256']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==trained['runtime_fingerprint']
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_port_semantics.py'),Path(__file__).with_name('paired_port_semantic_selection.py'),
        Path(__file__).with_name('port_semantic_verifier.py'),OUT/'head_full/report.json',weight,referencepath,REPO/'prototype/assembly_auto_review_robust_v3.py')}
    inputs={}
    for stage,count in (('inner',48),('outer',30)):
        reportpath=BASE/stage/'report.json';pins[str(reportpath)]=sha(reportpath);rows=load(reportpath)['cases'];assert len(rows)==count;inputs[stage]=[]
        for entry in rows:
            path=PROPOSALS/(stage+'_'+Path(entry['image']).stem+'_proposals.json');pins[str(path)]=sha(path);candidate=load(path)
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/entry['image'];pins[str(source)]=sha(source)
            inputs[stage].append(dict(entry=entry,candidate=candidate,source=source))
    destination.mkdir();save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,probability_gate=.98,contexts=[1.5,3.],
        proposals_two_distinct_weights_above_existing05_floor=True,inner48_then_outer30=True,no_GT_features_or_registration=True,
        GT_after_predictions_only=True,old_cues_and_shared5plus5_preserved=True,registration_failures_explicit_abstentions=True,
        validation_reused=True,no_automatic_deployment=True,field_accuracy=False))
    import cv2
    import numpy as np
    import torch
    import dino_feature_diff as dino
    import assembly_auto_review_robust_v3 as registration
    cv2.setNumThreads(2);torch.set_num_threads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
    loaded=torch.load(weight,map_location='cpu',weights_only=True);assert loaded['input_dimensions']==6144 and loaded['encoder_sha256']==trained['encoder_sha256']
    head=torch.nn.Linear(6144,3);head.load_state_dict(loaded['state_dict'],strict=True);head.requires_grad_(False).eval()
    reference=read_image(referencepath);cache=cached_alignments();started=time.monotonic()
    try:
        with CachedReferenceSIFT(reference):
            for stage,items in inputs.items():
                folder=destination/stage;folder.mkdir();records=[]
                for index,item in enumerate(items):
                    entry=item['entry'];name=entry['image'];candidate=item['candidate'];current=candidate['current'];proposals=candidate['candidates'];scores=[];valid_proposals=[]
                    status='no_candidate_exact_short_circuit';alignment=None
                    save(destination/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=len(items),seconds=round(time.monotonic()-started,2)))
                    if proposals:
                        image=read_image(item['source']);cached=cache.get(name)
                        if cached:
                            assert cached[1]['image_fingerprints']['source_sha256']==pins[str(item['source'])]
                            pins[str(cached[0])]=sha(cached[0]);alignment=copy.deepcopy(cached[1]['alignment'])
                        else:cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            valid_proposals=[proposals[i] for i in valid_boxes([p['box_xyxy'] for p in proposals],valid)]
                            boxes=[p['box_xyxy'] for p in valid_proposals]
                            with torch.inference_mode():
                                features=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                                scores=head(features).softmax(dim=1).tolist()
                            status='paired_inferred' if boxes else 'abstained_all_contexts_invalid'
                        else:status='abstained_registration'
                    trial=select(current,valid_proposals,scores,trained['checkpoint']['sha256'])
                    save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=valid_proposals,
                        probabilities=scores,feature_status=status,alignment=alignment,source_sha256=pins[str(item['source'])],head_sha256=trained['checkpoint']['sha256']))
                    targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                    old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
                    records.append(dict(image=name,current=metric(old,targets),trial=metric(new,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                        feature_status=status,proposals=len(proposals),valid_proposals=len(valid_proposals),additions=len(trial['paired_semantic_additions'])))
                    save(folder/'partial.json',dict(cases=records))
                totals={v:{k:sum(r[v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
                assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
                normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
                gain=totals['trial']['tp']>totals['current']['tp'] if stage=='inner' else totals['trial']['tp']>=totals['current']['tp']
                qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
                assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
                save(folder/'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
                    no_GT_in_model=True,validation_reused=True,no_model_deployment=True,field_accuracy=False))
                print(str(dict(stage=stage,qualifies=qualifies,summary=totals)),flush=True)
                if not qualifies:save(destination/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
        save(destination/'progress.json',dict(status='source_pairs_pass_requires_real_current_ROI_Qt_SAM_acceptance',no_automatic_deployment=True))
    except BaseException as error:
        save(destination/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
