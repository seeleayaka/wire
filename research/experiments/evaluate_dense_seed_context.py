"""Separate composite source test; standalone failed heads remain rejected."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,DATA,load,save,read_targets
sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True')
from inspection_agent.optional_port_crop_review import sha,read_image
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.teacher_student_port_support import STUDENT_SHA,TEACHER_SHA
from inspection_agent.port_tiling import near_artificial_edge
from dense_port_probe import frozen_features,preprocess
from dense_pixel_port_probe import DensePixelPortHead,decode
from dense_seed_context_policy import seeds_from_cases,context_windows,append_confirmed
from audit_port_multiscale_acceptance import metric,matches
OUT=ROOT/'artifacts/dense_seed_context_20261003'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'
TRAIN=ROOT/'artifacts/dense_pixel_port_probe_20261003/full'


def predict_entries(encoder,head,image,seeds,progress):
    import torch
    shape=list(image.shape[:2]);h,w=shape;windows=[]
    for seed in seeds:
        for window in context_windows(seed,shape):
            if window not in windows:windows.append(window)
    decoded={}
    with torch.inference_mode():
        for start in range(0,len(windows),2):
            batch=windows[start:start+2];crops=[image[y:b,x:r] for x,y,r,b in batch]
            features=dict(semantic=frozen_features(encoder,crops),rgb=torch.stack([preprocess(im) for im in crops]).to(torch.float16))
            outputs=decode(head(features),[im.shape[:2] for im in crops],score=.75,max_boxes=100)
            for window,rows in zip(batch,outputs):
                x,y,_,_=window;globalrows=[]
                for row in rows:
                    if near_artificial_edge(row['box_xyxy'],window,w,h,16):continue
                    l,t,r,b=row['box_xyxy'];globalrows.append(dict(row,box_xyxy=[l+x,t+y,r+x,b+y]))
                decoded[tuple(window)]=globalrows
            progress(window_completed=min(start+2,len(windows)),window_total=len(windows))
    return [dict(seed=copy.deepcopy(seed),windows=context_windows(seed,shape),
                 views=[decoded[tuple(w)] for w in context_windows(seed,shape)]) for seed in seeds]


def main():
    if OUT.exists():raise FileExistsError('Preserve prior experiment')
    report=load(TRAIN/'report.json');assert report['status']=='complete' and report['completed_epochs']==8 and report['frozen_encoder_unchanged']
    weight=Path(report['checkpoint']['path']);digest=sha(weight);assert digest==report['checkpoint']['sha256']=='2cbcc28430dae06b47ff2b7a8fd2631c92f18c7a4cedb14e45265950addfd034'
    encoderpath=REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth';assert sha(encoderpath)==report['encoder_sha256']
    frozen=resolution_runtime_fingerprint(REPO);assert frozen==report['runtime_fingerprint']
    paths=[Path(__file__),Path(__file__).with_name('dense_seed_context_policy.py'),Path(__file__).with_name('dense_pixel_port_probe.py'),
        Path(__file__).with_name('dense_port_probe.py'),Path(__file__).with_name('audit_port_multiscale_acceptance.py'),
        weight,encoderpath,TRAIN/'report.json',ROOT/'artifacts/dense_seed_context_preregistration_20261003/PLAN.md']
    pins={str(p):sha(p) for p in paths};inputs={};seed_sums={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        inputs[stage]=[];seed_sums[stage]=dict(images=count,seed_images=0,seeds=0)
        indexpath=PREP/stage/'index.json';pins[str(indexpath)]=sha(indexpath);records=load(indexpath)['records'];assert len(records)==count
        for row in records:
            path=Path(row['path']);assert sha(path)==row['sha256'];pins[str(path)]=row['sha256'];case=load(path)
            teacher,student=case['teacher'],case['student'];current=case['current']
            assert teacher['weight_sha256']==TEACHER_SHA and student['weight_sha256']==STUDENT_SHA
            assert teacher['image']==student['image']==row['image']
            source=DATA/'images'/('val01' if stage=='outer' else 'train01')/row['image'];pins[str(source)]=sha(source)
            assert teacher['source_sha256']==student['source_sha256']==pins[str(source)]
            seeds=seeds_from_cases(current,[teacher,student]);seed_sums[stage]['seeds']+=len(seeds);seed_sums[stage]['seed_images']+=int(bool(seeds))
            inputs[stage].append(dict(case=case,seeds=seeds,source=source))
    assert {p:sha(Path(p)) for p in pins}==pins
    OUT.mkdir();save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,head_sha256=digest,encoder_sha256=report['encoder_sha256'],
        standalone_head_crop_gate_still_rejected=True,new_composite_not_relabeled_standalone=True,
        score_both_views=.75,weak_seed_score=.25,seed_and_context_iou=.5,max_seeds=6,contexts=[640,960],seed_coverage=seed_sums,
        old_current_prefix_preserved=True,max_primary5_all_extra5=True,no_partial_feature_cache_in_seed_selector=True,
        labels_after_selection_only=True,all192_train_then48_inner_then30_outer=True,
        exact_no_seed_short_circuit=True,no_automatic_deployment=True,reference_pending=True,validation_reused=True,field_accuracy=False))
    import torch
    import dino_feature_diff as dino
    torch.set_num_threads(2);encoder=dino._model();encoder.requires_grad_(False).eval();torch.set_num_threads(2)
    saved=torch.load(weight,map_location='cpu',weights_only=True);assert saved['epochs']==8 and saved['grid']==56 and saved['input_size']==448
    head=DensePixelPortHead();head.load_state_dict(saved['state_dict'],strict=True);head.requires_grad_(False).eval();started=time.monotonic()
    try:
        for stage,items in inputs.items():
            folder=OUT/stage;folder.mkdir();records=[];fresh=0
            for index,item in enumerate(items):
                case=item['case'];entry=case['entry'];name=entry['image'];current=case['current'];seeds=item['seeds']
                def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,
                    completed=index,total=len(items),fresh_context_sources=fresh,seconds=round(time.monotonic()-started,2),**kw))
                if seeds:
                    progress(phase='fresh_DINO_RGB_context_decoding',window_completed=0,window_total=2*len(seeds))
                    entries=predict_entries(encoder,head,read_image(item['source']),seeds,lambda **kw:progress(phase='fresh_DINO_RGB_context_decoding',**kw));fresh+=1
                else:entries=[]
                trial=append_confirmed(current,entries,case['teacher']['predictions']['source_shape'],digest)
                save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,seeds=seeds,context_evidence=entries,
                    source_sha256=case['teacher']['source_sha256'],head_sha256=digest))
                targets=read_targets(stage,name,case['teacher']['predictions']['source_shape'],entry['label_sha256'],pins)
                old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
                records.append(dict(image=name,current=metric(old,targets),trial=metric(new,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                    seeds=len(seeds),additions=len(trial['dense_context_additions'])))
                save(folder/'partial.json',dict(cases=records));progress(phase='source_scored',completed_images=index+1)
            totals={v:{k:sum(r[v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            gain=totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']
            passed=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,fresh_context_sources=fresh,
                reference_pending=True,field_accuracy=False,production_changed=False))
            print(str(dict(stage=stage,qualifies=passed,summary=totals)),flush=True)
            if not passed:save(OUT/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
        save(OUT/'progress.json',dict(status='source_only_pass_requires_fresh_reference_GUI_SAM',no_automatic_deployment=True))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
