"""Fresh initial reports + mask-identity-verified SAM fusion + live port A/B."""
import os,sys,json,copy,time,shutil
from pathlib import Path
from dataclasses import replace
sys.dont_write_bytecode=True
WORK=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OLD=WORK/'artifacts/main_scenario_live_20261002'
OUT=WORK/'artifacts/rescue_mainline_ab_20261002_v2'
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'),YOLO_CONFIG_DIR=str(OUT/'model_config'))
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'prototype'))
import torch
torch.set_num_threads(4)
import cv2,numpy as np
import assembly_auto_review_dino_v2 as entry
gui=entry.implementation
from inspection_agent.optional_port_crop_review import sha,SCENE
from inspection_agent.independent_port_rescue import run_independent_port_rescue
from inspection_agent.port_crop_gui_bridge import render_port_overlay
from inspection_agent.gui_bridge import create_task_from_visual_report

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def bbox(row):return row.get('bbox_xyxy') or [row[k] for k in ('left','top','right','bottom')]
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0
def metrics(boxes,targets):
    best=[max((iou(b,t) for b in boxes),default=0) for t in targets]
    return dict(candidate_count=len(boxes),target_count=len(targets),target_any_overlap=sum(v>0 for v in best),
        target_iou050=sum(v>=.5 for v in best),candidate_iou050=sum(any(iou(b,t)>=.5 for t in targets) for b in boxes))

def main():
    if OUT.exists():raise FileExistsError('Do not overwrite a prior A/B run')
    OUT.mkdir();(OUT/'model_config/Ultralytics').mkdir(parents=True)
    old_manifest=load(OLD/'manifest.json');prior=load(OLD/'report.json')['cases']
    ref=Path(old_manifest['reference']);assert sha(ref)==old_manifest['reference_sha256']
    for row in old_manifest['images']:assert sha(row['path'])==row['sha256']
    settings=gui.sam3_wire_fusion.load_settings()
    assert sha(settings.checkpoint)==old_manifest['weights']['sam3']
    assert sha(REPO/'prototype/sam3_wire_fusion.py')==old_manifest['source_sha256'][str(REPO/'prototype/sam3_wire_fusion.py')]
    assert sha(REPO/'models/dinov2/weights/dinov2_vits14_pretrain.pth')==old_manifest['weights']['dino']
    sources=[REPO/'prototype/assembly_auto_review_dino.py',REPO/'prototype/sam3_wire_fusion.py',
        REPO/'inspection_agent/independent_port_rescue.py',REPO/'prototype/tiled_dino_review.py']
    source_hashes={str(p):sha(p) for p in sources}
    records=[];protected={};began=time.monotonic()
    reference_key=gui.sam3_wire_fusion._fingerprint(ref)
    old_ref=OLD/'sam_cache/reference_masks'/reference_key
    ref_report=load(old_ref/'report.json')
    assert ref_report['prompt']==settings.prompt and ref_report['confidence_threshold']==settings.confidence_threshold
    new_ref=OUT/'sam_cache/reference_masks'/reference_key;new_ref.mkdir(parents=True)
    for p in old_ref.iterdir():
        if p.suffix in ('.png','.json'):shutil.copy2(p,new_ref/p.name);protected[str(p)]=sha(p)
    active_settings=replace(settings,cache_dir=OUT/'sam_cache')
    original_loader=gui.sam3_wire_fusion.load_settings
    gui.sam3_wire_fusion.load_settings=lambda *args,**kwargs:active_settings
    original_out=gui.adaptive.robust.auto.base.OUT
    save(OUT/'manifest.json',dict(images=old_manifest['images'],reference=str(ref),roi=old_manifest['roi'],
        selection='same four fixed pilot cases; no parameter tuning',source_hashes=source_hashes,
        fresh_initial_and_ports=True,sam_raw_masks_reused_only_on_byte_identity=True,labels_used_for_inference=False))
    try:
        for row in prior:
            start=time.monotonic();old_output=Path(row['output']);path=Path(old_manifest['images'][len(records)]['path'])
            assert path.name==row['image']
            case=OUT/path.stem;case.mkdir();gui.adaptive.robust.auto.base.OUT=case
            protected[str(old_output/'report.json')]=sha(old_output/'report.json')
            protected[row['task']]=sha(row['task'])
            cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(ref,path,old_manifest['roi'])
            payload=[];worker.completed.connect(payload.append)
            print('INITIAL '+path.name,flush=True);worker.run()
            assert len(payload)==1 and payload[0]['status']=='ready_for_sam3',payload
            initial=payload[0];output=Path(initial['output']);report=initial['report']
            save(output/'initial_report.json',report)
            assert report['analysis_check_rois']==old_manifest['roi']
            # Never transplant a mask onto a different alignment, even for the same source.
            assert sha(output/'aligned.jpg')==sha(old_output/'aligned.jpg'),'Aligned bytes differ; SAM reuse refused'
            assert sha(output/'valid_warp_mask.png')==sha(old_output/'valid_warp_mask.png')
            original_probe=old_output/'sam3/inspection';probe=load(original_probe/'report.json')
            assert Path(probe['input'])==old_output/'aligned.jpg'
            assert probe['prompt']==settings.prompt and probe['confidence_threshold']==settings.confidence_threshold
            new_probe=output/'sam3/inspection';new_probe.mkdir(parents=True)
            for p in original_probe.iterdir():
                if p.suffix in ('.png','.jpg','.json'):
                    shutil.copy2(p,new_probe/p.name);protected[str(p)]=sha(p)
            probe['input']=str(output/'aligned.jpg');save(new_probe/'report.json',probe)
            fusion_worker=gui.Sam3FusionWorker(ref,output/'aligned.jpg',output/'valid_warp_mask.png',
                initial['dino_regions'],output/'check_heatmap.jpg',output)
            fused=[];fusion_worker.completed.connect(fused.append);fusion_worker.run()
            assert len(fused)==1 and fused[0]['status']=='ok',fused
            fusion=fused[0]
            assert fusion['reference_sam3']['cache_hit'] and fusion['inspection_sam3']['cache_hit']
            report.update(review_regions=fusion['review_regions'],sam3_fusion=fusion,
                decision='possible_difference_manual_review' if fusion['review_regions'] else 'no_significant_wire_related_difference')
            save(output/'report_off.json',report);before=copy.deepcopy(report)
            task=create_task_from_visual_report(output/'agent_task.json',task_id='rescue-ab-'+path.stem,
                scene_type=SCENE,reference=str(ref),inspection=str(path),visual_report=report,source_report=output/'report_off.json')
            task_sha=sha(output/'agent_task.json');assert task.state=='awaiting_human_review'
            off=run_independent_port_rescue(report,project=REPO,enabled=False,scene=SCENE)
            on=run_independent_port_rescue(report,project=REPO,enabled=True,scene=SCENE)
            assert off['status']=='disabled' and report==before and on['parents']==report['review_regions']
            assert len(on['rescue_hints'])<=1 and not on['automatic_fault_verdict']
            assert sha(output/'agent_task.json')==task_sha
            save(output/'rescue_evidence.json',on)
            augmented=copy.deepcopy(report);augmented['independent_port_rescue']=on
            save(output/'report_on.json',augmented)
            if on['status']=='applied':render_port_overlay(output/'aligned.jpg',
                dict(parents=on['parents'],tile_hints=on['rescue_hints']),output/'rescue_overlay.jpg')
            records.append(dict(image=path.name,output=str(output),status=on['status'],reason=on['fallback_reason'],
                new_hints=len(on['rescue_hints']),seconds=round(time.monotonic()-start,2),sam_mask_cache_identity_verified=True,
                original_candidates_unchanged=True,task_unchanged=True))
            save(OUT/'progress.json',dict(cases=records));print('DONE '+json.dumps(records[-1]),flush=True)
    finally:
        gui.sam3_wire_fusion.load_settings=original_loader;gui.adaptive.robust.auto.base.OUT=original_out
    # Labels only after all inference and selection is finalized.
    data=Path(ref).parents[2]
    for record in records:
        output=Path(record['output']);report=load(output/'report_off.json');evidence=load(output/'rescue_evidence.json')
        label=data/'labels/test01'/(Path(record['image']).stem+'.txt')
        targets=[];matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
        for line in label.read_text(encoding='utf-8').splitlines():
            _,cx,cy,w,h=map(float,line.split());l=(cx-w/2)*3648;t=(cy-h/2)*2736;r=(cx+w/2)*3648;b=(cy+h/2)*2736
            q=cv2.perspectiveTransform(np.float32([[[round(l),round(t)],[round(r),round(t)],
                [round(r),round(b)],[round(l),round(b)]]]),matrix).reshape(-1,2)
            targets.append([int(np.floor(q[:,0].min())),int(np.floor(q[:,1].min())),int(np.ceil(q[:,0].max())),int(np.ceil(q[:,1].max()))])
        baseline=[bbox(c) for c in report['review_regions']];extra=[bbox(h['box']) for h in evidence['rescue_hints']]
        record.update(off=metrics(baseline,targets),on=metrics(baseline+extra,targets),added=metrics(extra,targets),label_sha256=sha(label))
        save(output/'evaluation.json',dict(aligned_targets=targets,comparison=record,labels_not_human_confirmation=True))
    assert {p:sha(p) for p in protected}==protected
    assert {str(p):sha(p) for p in sources}==source_hashes
    save(OUT/'report.json',dict(status='complete',cases=records,seconds=round(time.monotonic()-began,2),
        old_artifacts_unchanged=True,source_unchanged=True,labels_used_for_inference=False,
        fresh_initial_and_ports=True,sam_raw_inference=False,sam_fusion_recomputed=True,field_accuracy_claimed=False))
    print('AB COMPLETE',flush=True)

if __name__=='__main__':main()
