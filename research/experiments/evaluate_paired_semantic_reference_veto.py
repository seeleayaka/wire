"""Layered full-head composite source gates; source cached, reference fresh."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_semantic_full_train import NEW
from prepare_paired_port_semantics import ROOT,REPO,OUT,DATA,PROPOSALS,load,save,sha,read_image,cached_alignments,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from paired_port_semantic_selection import select
from paired_semantic_reference_veto import veto
from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
from port_semantic_verifier import embeddings
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.teacher_student_port_support import TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.optional_port_crop_review import predict,aligned_predictions
from inspection_agent.context_port_recheck import predict_seed_views,complete
sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
                  YOLO_CONFIG_DIR=str(NEW/'yolo_config'))


def main():
    destination=NEW/'reference_gates_v2'
    if destination.exists():raise FileExistsError('Preserve reference gates')
    fulltrain=load(NEW/'full_train/report.json');assert fulltrain['status']=='complete'
    trained=load(OUT/'head_full/report.json');weight=Path(trained['checkpoint']['path']);digest=sha(weight)
    assert digest==trained['checkpoint']['sha256']==fulltrain['head_sha256']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==trained['runtime_fingerprint']
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins=dict(fulltrain['pins'])
    for p in (Path(__file__),Path(__file__).with_name('paired_semantic_reference_veto.py'),Path(__file__).with_name('paired_port_semantic_selection.py'),
              NEW/'full_train/report.json',referencepath,REPO/TEACHER_RELATIVE,REPO/STUDENT_RELATIVE,
              ROOT/'artifacts/paired_semantic_reference_veto_preregistration_20261003/PLAN.md'):
        pins[str(p)]=sha(p)
    assert pins[str(REPO/TEACHER_RELATIVE)]==TEACHER_SHA and pins[str(REPO/STUDENT_RELATIVE)]==STUDENT_SHA
    inputs={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        reportpath=BASE/stage/'report.json';pins[str(reportpath)]=sha(reportpath)
        entries=load(reportpath)['cases'];assert len(entries)==count
        inputs[stage]=entries
        for entry in entries:
            name=entry['image'];path=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
            pins[str(path)]=sha(path)
            if stage!='outer':
                path=(NEW/'full_train' if stage=='train' else OUT/'holdouts/inner')/(Path(name).stem+'_predictions.json')
            else:path=PROPOSALS/('outer_'+Path(name).stem+'_proposals.json')
            pins[str(path)]=sha(path)
    assert {p:sha(Path(p)) for p in pins}==pins
    destination.mkdir(parents=True)
    save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,old_native_prefix_preserved=True,
        source_train_and_inner_cached=True,new_reference_full_and_matched_views=True,probability=.98,
        reference_score=.25,reference_same_class_iou=.5,no_GT_policy_inputs=True,source_gates_then_outer=True,
        source_composite_only_real_current_ROI_Qt_SAM_pending=True,no_deployment=True,validation_reused=True,field_accuracy=False))
    import cv2
    import numpy as np
    import torch
    (NEW/'yolo_config').mkdir(parents=True,exist_ok=True)
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(2);reference=read_image(referencepath);models=[];base_refs=[];started=time.monotonic()
    def progress(**kw):save(destination/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    for model_sha,relative in ((TEACHER_SHA,TEACHER_RELATIVE),(STUDENT_SHA,STUDENT_RELATIVE)):
        progress(phase='fresh_reference_full',checkpoint_sha256=model_sha)
        native=YOLO(str(REPO/relative));assert native.task=='segment' and dict(native.names)=={0:'unplugged_plug',1:'unplugged_jack'}
        class Capped:
            def __init__(self,model):self.model=model
            def predict(self,*args,**kwargs):
                torch.set_num_threads(4);value=self.model.predict(*args,**kwargs);torch.set_num_threads(4);return value
        model=Capped(native);models.append((model_sha,model));raw=predict(model,reference)
        save(destination/(model_sha[:8]+'_reference_full.json'),dict(reference_sha256=REFERENCE_SHA,weight_sha256=model_sha,predictions=raw))
        base_refs.extend(aligned_predictions(raw['merged_predictions'],np.eye(3),reference.shape[:2],reference.shape[:2]))
    encoder=None;head=None;alignment_cache=cached_alignments()
    with CachedReferenceSIFT(reference):
        for stage,entries in inputs.items():
            folder=destination/stage;folder.mkdir();records=[];fresh_candidate_sources=0
            for index,entry in enumerate(entries):
                name=entry['image'];progress(stage=stage,image=name,completed=index,total=len(entries),phase='candidate_reference_views')
                status='cached_source_semantic';alignment=None
                if stage!='outer':
                    path=(NEW/'full_train' if stage=='train' else OUT/'holdouts/inner')/(Path(name).stem+'_predictions.json')
                    cached=load(path);current=cached['current'];trial=cached['trial'];alignment=cached['alignment']
                else:
                    candidate=load(PROPOSALS/('outer_'+Path(name).stem+'_proposals.json'));current=candidate['current'];native=candidate['candidates']
                    probabilities=[];valid_proposals=[];status='no_candidate_exact_short_circuit'
                    if native:
                        source=DATA/'images/val01'/name;image=read_image(source);cached=alignment_cache.get(name)
                        if cached:
                            assert cached[1]['image_fingerprints']['source_sha256']==pins[str(source)]
                            pins[str(cached[0])]=sha(cached[0]);alignment=copy.deepcopy(cached[1]['alignment'])
                        else:
                            import assembly_auto_review_robust_v3 as registration
                            cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            valid_proposals=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],valid)]
                            if encoder is None:
                                import dino_feature_diff as dino
                                torch.set_num_threads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
                                loaded=torch.load(weight,map_location='cpu',weights_only=True);head=torch.nn.Linear(6144,3)
                                head.load_state_dict(loaded['state_dict']);head.requires_grad_(False).eval()
                            boxes=[p['box_xyxy'] for p in valid_proposals]
                            with torch.inference_mode():probabilities=head(paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))).softmax(dim=1).tolist()
                            status='fresh_paired_source' if boxes else 'abstained_invalid_context'
                        else:status='abstained_registration'
                    trial=select(current,valid_proposals,probabilities,digest)
                candidates=trial['paired_semantic_additions'];accepted=[];audits=[];reference_views=[]
                if candidates:
                    assert alignment['alignment_quality']['reliable']
                    matrix=np.asarray(alignment['source_to_reference_homography'],dtype=np.float64)
                    native=copy.deepcopy(candidates)
                    for row in native:row['support_tiles']=[]
                    mapped=aligned_predictions(native,matrix,[2736,3648],reference.shape[:2])
                    seeds=[dict(class_id=p['class_id'],confidence=p['confidence'],box_xyxy=[p[k] for k in ('left','top','right','bottom')]) for p in mapped]
                    refs=copy.deepcopy(base_refs)
                    for model_sha,model in models:
                        views=predict_seed_views(model,reference,seeds);reference_views.append(dict(weight_sha256=model_sha,views=views))
                        for record in views:
                            for view in record['views']:
                                for p in view:
                                    if p['confidence']>.25 and complete(p,reference.shape[:2]):
                                        refs.append(dict(zip(('left','top','right','bottom'),p['box_xyxy']),class_id=p['class_id'],confidence=p['confidence'],valid_warp_fraction=1.,support_tiles=[]))
                    accepted,audits=veto(candidates,refs,matrix,[2736,3648],reference.shape[:2]);fresh_candidate_sources+=1
                selected=copy.deepcopy(current['all_predictions'])+accepted
                assert selected[:len(current['all_predictions'])]==current['all_predictions'] and len(selected)<=len(current['primary'])+5
                save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial_native=selected,candidates=candidates,
                    accepted=accepted,reference_veto_audit=audits,reference_views=reference_views,alignment=alignment,feature_status=status,head_sha256=digest))
                targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                old=current['all_predictions'];oh,nh=matches(old,targets)[0],matches(selected,targets)[0]
                records.append(dict(image=name,current=metric(old,targets),trial=metric(selected,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                                    candidates=len(candidates),accepted=len(accepted),vetoed=len(candidates)-len(accepted),feature_status=status))
                save(folder/'partial.json',dict(cases=records))
            totals={v:{k:sum(r[v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            gain=totals['trial']['tp']>=totals['current']['tp'] if stage=='outer' else totals['trial']['tp']>totals['current']['tp']
            qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
                fresh_reference_candidate_sources=fresh_candidate_sources,validation_reused=True,current_ROI_Qt_SAM_pending=True,no_deployment=True,field_accuracy=False))
            print(str(dict(stage=stage,qualifies=qualifies,summary=totals)),flush=True)
            if not qualifies:save(destination/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
    save(destination/'progress.json',dict(status='source_composite_pass_requires_real_ROI_Qt_SAM_acceptance',no_deployment=True))


if __name__=='__main__':main()
