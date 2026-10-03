"""Existing committee native output + fresh original reference-only veto."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from paired_semantic_reference_veto import veto
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.teacher_student_port_support import TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.optional_port_crop_review import predict,aligned_predictions
from inspection_agent.context_port_recheck import predict_seed_views,complete
from audit_port_multiscale_acceptance import metric,matches
SOURCE=ROOT/'artifacts/paired_semantic_committee_20261003'
OUT=ROOT/'artifacts/paired_committee_reference_veto_20261003'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'yolo_config'))


def main():
    if OUT.exists():raise FileExistsError('Preserve committee-reference experiment')
    protocol=load(SOURCE/'protocol.json');assert {p:sha(Path(p)) for p in protocol['pins']}==protocol['pins']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==protocol['runtime_fingerprint']
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins=dict(protocol['pins']);inputs={}
    for p in (Path(__file__),Path(__file__).with_name('paired_semantic_reference_veto.py'),ROOT/'artifacts/paired_committee_reference_preregistration_20261003/PLAN.md',
              REPO/TEACHER_RELATIVE,REPO/STUDENT_RELATIVE,referencepath):pins[str(p)]=sha(p)
    assert pins[str(REPO/TEACHER_RELATIVE)]==TEACHER_SHA and pins[str(REPO/STUDENT_RELATIVE)]==STUDENT_SHA
    for stage,count in (('train',192),('inner',48),('outer',30)):
        path=SOURCE/stage/'report.json';pins[str(path)]=sha(path);report=load(path)
        assert report['status']=='complete' and len(report['cases'])==count
        baselinepath=BASE/stage/'report.json';pins[str(baselinepath)]=sha(baselinepath)
        entries={row['image']:row for row in load(baselinepath)['cases']};inputs[stage]=[]
        for row in report['cases']:
            name=row['image'];path=SOURCE/stage/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path)
            imagepath=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(imagepath)]=sha(imagepath)
            inputs[stage].append((entries[name],path))
    assert {p:sha(Path(p)) for p in pins}==pins
    OUT.mkdir();(OUT/'yolo_config').mkdir()
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,all270_source_cases=True,
        source_committee_cached=True,fresh_reference_full_and_matched_views=True,
        reference_score=.25,reference_same_class_iou=.5,no_backfill_after_veto=True,
        no_GT_filter_inputs=True,training_committee_not_OOF=True,validation_reused=True,
        current_ROI_Qt_SAM_pending=True,no_deployment=True,field_accuracy=False))
    import torch
    import cv2
    import numpy as np
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(2);reference=read_image(referencepath);models=[];base_refs=[];started=time.monotonic()
    def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    for digest,relative in ((TEACHER_SHA,TEACHER_RELATIVE),(STUDENT_SHA,STUDENT_RELATIVE)):
        progress(phase='fresh_reference_full',weight_sha256=digest)
        native=YOLO(str(REPO/relative));assert native.task=='segment' and dict(native.names)=={0:'unplugged_plug',1:'unplugged_jack'}
        class Capped:
            def __init__(self,model):self.model=model
            def predict(self,*a,**kw):
                torch.set_num_threads(4);value=self.model.predict(*a,**kw);torch.set_num_threads(4);return value
        model=Capped(native);models.append((digest,model));raw=predict(model,reference)
        save(OUT/(digest[:8]+'_reference_full.json'),dict(reference_sha256=REFERENCE_SHA,weight_sha256=digest,predictions=raw))
        base_refs.extend(aligned_predictions(raw['merged_predictions'],np.eye(3),reference.shape[:2],reference.shape[:2]))
    try:
        for stage,entries in inputs.items():
            folder=OUT/stage;folder.mkdir();records=[];fresh=0
            for index,(entry,path) in enumerate(entries):
                name=entry['image'];cached=load(path);current=cached['current'];candidates=cached['trial']['paired_committee_additions']
                alignment=cached['alignment'];accepted=[];audits=[];views_evidence=[]
                progress(stage=stage,image=name,completed=index,total=len(entries),phase='fresh_reference_candidate_views')
                if candidates:
                    assert alignment['alignment_quality']['reliable']
                    matrix=np.asarray(alignment['source_to_reference_homography'],dtype=np.float64);native=copy.deepcopy(candidates)
                    for row in native:row['support_tiles']=[]
                    mapped=aligned_predictions(native,matrix,[2736,3648],reference.shape[:2])
                    seeds=[dict(class_id=row['class_id'],confidence=row['confidence'],box_xyxy=[row[k] for k in ('left','top','right','bottom')]) for row in mapped]
                    refs=copy.deepcopy(base_refs)
                    for digest,model in models:
                        views=predict_seed_views(model,reference,seeds);views_evidence.append(dict(weight_sha256=digest,views=views))
                        for record in views:
                            for view in record['views']:
                                for row in view:
                                    if row['confidence']>.25 and complete(row,reference.shape[:2]):
                                        refs.append(dict(zip(('left','top','right','bottom'),row['box_xyxy']),class_id=row['class_id'],confidence=row['confidence'],valid_warp_fraction=1.,support_tiles=[]))
                    accepted,audits=veto(candidates,refs,matrix,[2736,3648],reference.shape[:2]);fresh+=1
                selected=copy.deepcopy(current['all_predictions'])+accepted
                assert selected[:len(current['all_predictions'])]==current['all_predictions'] and len(selected)<=len(current['primary'])+5
                save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial_native=selected,
                    candidates=candidates,accepted=accepted,reference_veto_audit=audits,reference_views=views_evidence,alignment=alignment))
                targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                old=current['all_predictions'];oh,nh=matches(old,targets)[0],matches(selected,targets)[0]
                records.append(dict(image=name,current=metric(old,targets),trial=metric(selected,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                    candidates=len(candidates),accepted=len(accepted),vetoed=len(candidates)-len(accepted)))
                save(folder/'partial.json',dict(cases=records))
            totals={v:{k:sum(row[v][k] for row in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(row['trial']['predictions'] for row in records if row['image'].startswith('normal_'))
            gain=totals['trial']['tp']>=totals['current']['tp'] if stage=='outer' else totals['trial']['tp']>totals['current']['tp']
            qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(row['lost'] for row in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
                fresh_reference_candidate_sources=fresh,source_committee_cached=True,validation_reused=True,
                current_ROI_Qt_SAM_pending=True,no_deployment=True,field_accuracy=False))
            print(str(dict(stage=stage,qualifies=qualifies,summary=totals)),flush=True)
            if not qualifies:progress(status='rejected',stage=stage,summary=totals);return
        progress(status='source_composite_pass_requires_real_ROI_Qt_SAM_acceptance',no_deployment=True)
    except BaseException as error:
        progress(status='failed',error=type(error).__name__+': '+str(error));raise


if __name__=='__main__':main()
