"""Fresh four-image acceptance pilot using the actual desktop workers.

No tuning, no source edits, no operator conclusions, no repair/reinspection claims.
Labels are opened only after all predictions and task creation have completed.
"""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
import time
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/main_scenario_live_20261002'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'prototype'))
import torch
torch.set_num_threads(4)
import cv2
import numpy as np
import assembly_auto_review_dino_v2 as entry
app=entry.implementation
import tiled_dino_review as tiled
from inspection_agent import InspectionTask, WorkflowError
from inspection_agent.gui_bridge import create_task_from_visual_report

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):digest.update(part)
    return digest.hexdigest()

def box(row):
    return row.get('bbox_xyxy') or [row[k] for k in ('left','top','right','bottom')]

def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union else 0.0

def main():
    if OUT.exists():raise FileExistsError('Fresh output required; do not overwrite prior run')
    OUT.mkdir(parents=True)
    ref=DATA/'images/train01/normal_073.JPG'
    paths=[sorted((DATA/'images/test01').glob(kind+'_*.JPG'))[0] for kind in ('normal','damaged','disconnected','misrouted')]
    source_files=[REPO/'prototype/assembly_auto_review_dino.py',REPO/'prototype/tiled_dino_review.py',REPO/'prototype/sam3_wire_fusion.py',REPO/'prototype/assembly_auto_review_robust_v3.py']
    originals={str(p):sha(p) for p in source_files}
    selected=[{'image':p.name,'path':str(p),'sha256':sha(p)} for p in paths]
    settings=app.sam3_wire_fusion.load_settings()
    settings=replace(settings,cache_dir=OUT/'sam_cache')
    app.sam3_wire_fusion.validate_resources(settings)
    loader=app.sam3_wire_fusion.load_settings
    app.sam3_wire_fusion.load_settings=lambda *args,**kwargs:settings
    manifest=dict(selection='first alphabetically per filename category; fixed before inference',reference=str(ref),reference_sha256=sha(ref),images=selected,source_sha256=originals,roi=[[.03,.04,.97,.96]],candidate_budget=tiled.LARGE_ROI_MAX_CANDIDATES,weights={'dino':sha(REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth'),'sam3':sha(settings.checkpoint)},new_inference=True,labels_used_for_inference=False,optional_port_model_enabled=False)
    save(OUT/'manifest.json',manifest)
    records=[];started=time.perf_counter();old_out=app.adaptive.robust.auto.base.OUT
    try:
        for path in paths:
            begin=time.perf_counter();case=OUT/path.stem;case.mkdir()
            app.adaptive.robust.auto.base.OUT=case
            cv2.setRNGSeed(0)
            worker=app.InitialReviewWorker(ref,path,manifest['roi'])
            worker.stage_changed.connect(lambda text:print(text,flush=True))
            payload=[];worker.completed.connect(payload.append)
            print('START '+path.name,flush=True);worker.run()
            if len(payload)!=1:raise RuntimeError('No worker completion payload')
            result=payload[0];save(case/'initial_worker_payload.json',result)
            if result['status']=='error':
                records.append(dict(image=path.name,status='error',error=result['error']));continue
            output=Path(result['output']);report=result['report']
            save(output/'initial_report.json',report)
            if result['status']=='ready_for_sam3':
                print('SAM3 '+path.name,flush=True)
                sam_worker=app.Sam3FusionWorker(ref,output/'aligned.jpg',output/'valid_warp_mask.png',result['dino_regions'],output/'check_heatmap.jpg',output)
                fusion=[];sam_worker.completed.connect(fusion.append);sam_worker.run()
                if len(fusion)!=1:raise RuntimeError('No SAM completion payload')
                fusion=fusion[0]
                if fusion.get('status')=='ok':
                    regions=fusion['review_regions'];report.update(decision='possible_difference_manual_review' if regions else 'no_significant_wire_related_difference',review_regions=regions,sam3_fusion=fusion)
                else:
                    regions=result['dino_regions'];report.update(decision='possible_difference_manual_review' if regions else 'no_significant_difference',review_regions=regions,sam3_fusion={'status':'error','error':fusion.get('error'),'fallback':'Retained original DINO candidates'})
                save(output/'sam_worker_result.json',fusion)
            save(output/'report.json',report)
            task_path=output/'agent_task.json'
            task=create_task_from_visual_report(task_path,task_id='live-pilot-'+path.stem,scene_type='mendeley_pc_wiring_same_camera',reference=str(ref),inspection=str(path),visual_report=report,source_report=output/'report.json')
            reloaded=InspectionTask.load(task_path)
            assert reloaded.state=='awaiting_human_review'
            assert not reloaded.to_report()['human_conclusions']
            refused=False
            try:reloaded.add_repair_guidance('UNAUTHORIZED TEST MUST FAIL',evidence_ids=['1'])
            except WorkflowError:refused=True
            assert refused
            assert InspectionTask.load(task_path).to_report()==task.to_report()
            row=dict(image=path.name,status=result['status'],output=str(output),task=str(task_path),seconds=round(time.perf_counter()-begin,2),alignment_reliable=report.get('alignment_quality',{}).get('reliable') is True,dino_fallback=any('dino_error' in x for x in report.get('local_alignment',[])),dino_candidates=len(report.get('dino_review_regions',[])),fusion_candidates=len(report.get('review_regions',[])),sam_status=report.get('sam3_fusion',{}).get('status'),human_confirmation=False,repair_guidance_blocked=True,reinspection_performed=False)
            records.append(row);save(OUT/'progress.json',{'cases':records,'elapsed_seconds':round(time.perf_counter()-started,2)})
            print('DONE '+json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        app.adaptive.robust.auto.base.OUT=old_out;app.sam3_wire_fusion.load_settings=loader
    # Labels first opened now, after detection and persisted task evidence.
    for row in records:
        if row['status']=='error':continue
        report=json.loads((Path(row['output'])/'report.json').read_text(encoding='utf-8'))
        path=DATA/'images/test01'/row['image'];image=app.adaptive.robust.auto.base.read_image(path)
        targets=[]
        label_path=DATA/'labels/test01'/(path.stem+'.txt')
        for line in label_path.read_text(encoding='utf-8').splitlines():
            _,cx,cy,w,h=map(float,line.split());iw,ih=image.shape[1],image.shape[0]
            targets.append([round((cx-w/2)*iw),round((cy-h/2)*ih),round((cx+w/2)*iw),round((cy+h/2)*ih)])
        transform=report.get('alignment',{}).get('source_to_reference_homography')
        if transform is None:
            row['evaluation_unavailable']='No accepted transform';continue
        warped=[]
        for l,t,r,b in targets:
            q=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),np.array(transform)).reshape(-1,2)
            warped.append([int(np.floor(q[:,0].min())),int(np.floor(q[:,1].min())),int(np.ceil(q[:,0].max())),int(np.ceil(q[:,1].max()))])
        stages={}
        for name,key in [('dino','dino_review_regions'),('fusion','review_regions')]:
            predictions=[box(c) for c in report.get(key,[])];ious=[max((overlap(c,t) for c in predictions),default=0) for t in warped]
            stages[name]=dict(target_count=len(warped),target_any_overlap=sum(v>0 for v in ious),target_iou025=sum(v>=.25 for v in ious),target_iou050=sum(v>=.5 for v in ious),candidate_count=len(predictions),candidate_any_overlap=sum(any(overlap(c,t)>0 for t in warped) for c in predictions),target_best_ious=ious)
        evaluation=dict(original_targets=targets,aligned_targets=warped,label_sha256=sha(label_path),stages=stages,strict_precision_claimed=False,labels_not_operator_confirmation=True)
        save(Path(row['output'])/'label_audit.json',evaluation);row['evaluation']=evaluation
        aligned=app.adaptive.robust.auto.base.read_image(Path(row['output'])/'aligned.jpg')
        canvas=aligned.copy()
        for idx,t in enumerate(warped,1):
            cv2.rectangle(canvas,(t[0],t[1]),(t[2],t[3]),(255,230,0),3)
            cv2.putText(canvas,'T'+str(idx),(t[0],max(28,t[1])),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,230,0),2)
        for idx,c in enumerate(report.get('review_regions',[]),1):
            l,t,r,b=map(int,box(c));cv2.rectangle(canvas,(l,t),(r,b),(0,165,255),4)
            cv2.putText(canvas,'C'+str(idx),(l,max(30,t-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,165,255),2)
        app.adaptive.robust.auto.base.write_image(Path(row['output'])/'evaluation_overlay.jpg',canvas)
    assert {str(p):sha(p) for p in source_files}==originals
    assert [{'image':p.name,'path':str(p),'sha256':sha(p)} for p in paths]==selected
    save(OUT/'report.json',dict(status='complete',elapsed_seconds=round(time.perf_counter()-started,2),cases=records,manifest=manifest,source_unchanged=True,labels_used_for_inference=False,field_accuracy_claimed=False,human_confirmation=False,reinspection_performed=False))
    print('PILOT COMPLETE '+str(OUT/'report.json'),flush=True)

if __name__=='__main__':main()
