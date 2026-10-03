"""Evaluate new rescue gates with pinned source predictions and fresh registration."""
import os
import sys
import copy
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/independent_port_rescue_20261002'
OLD=ROOT/'artifacts/port_bridge_fix_20261002/independent_test01'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'prototype'))
os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
import torch
import cv2
import numpy as np
import assembly_auto_review_robust_v3 as registration
from inspection_agent.optional_port_crop_review import read_image,sha,predict,REFERENCE_SHA,aligned_predictions
from inspection_agent.port_tiling import box_iou
from independent_port_rescue import run_rescue

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if OUT.exists():raise FileExistsError('Keep prior output')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    reference=DATA/'images/train01/normal_073.JPG';ref=read_image(reference)
    manifest=load(OLD/'manifest.json');pins={p['name']:p['sha256'] for p in manifest['images']}
    weight=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
    assert sha(reference)==REFERENCE_SHA and sha(weight)==manifest['weight_sha256']
    from ultralytics import YOLO
    torch.set_num_threads(4);torch.manual_seed(20260929)
    model=YOLO(str(weight));reference_predictions=predict(model,ref)
    reference_predictions['source_sha256']=sha(reference)
    save(OUT/'reference_predictions.json',reference_predictions)
    roi=dict(left=3648*.03,top=2736*.04,right=3648*.97,bottom=2736*.96)
    cases=[];start=time.perf_counter()
    for idx,name in enumerate(sorted(pins),1):
        source=DATA/'images/test01'/name;source_sha=sha(source);assert source_sha==pins[name]
        image=read_image(source);cv2.setRNGSeed(0)
        aligned,geometry=registration.automatic_homography(ref,image)
        assert sha(source)==source_sha
        report=dict(reference=str(reference),inspection=str(source),review_regions=[],existing_port_hints=[],
            alignment_quality=geometry.get('alignment_quality',{}),alignment=geometry,local_alignment=[],
            image_fingerprints=dict(source_sha256=source_sha,reference_sha256=REFERENCE_SHA,stable_during_visual_analysis=True))
        cached=load(OLD/(Path(name).stem+'_predictions.json'))['predictions']
        result=run_rescue(report,project=REPO,reference_predictions=reference_predictions,roi=roi,enabled=True,
            scene='mendeley_pc_wiring_same_camera',prediction_provider=lambda image,c=cached:copy.deepcopy(c))
        case=dict(image=name,geometry=geometry,result=result,source_sha256=source_sha,
                  cached_prediction_sha256=sha(OLD/(Path(name).stem+'_predictions.json')))
        save(OUT/(Path(name).stem+'_rescue.json'),case);cases.append(case)
        print(f"CASE {idx}/30 {name} {result['status']} hints={len(result['rescue_hints'])}",flush=True)
        if aligned is not None:
            ok,encoded=cv2.imencode('.jpg',aligned);assert ok;encoded.tofile(str(OUT/(Path(name).stem+'_aligned.jpg')))
    groups={}
    for case in cases:
        name=case['image'];targets=[]
        for line in (DATA/'labels/test01'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').splitlines():
            _,cx,cy,w,h=map(float,line.split())
            targets.append([(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736])
        matrix=case['geometry'].get('source_to_reference_homography')
        warped=[]
        if matrix is not None:
            for l,t,r,b in targets:
                p=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),np.asarray(matrix)).reshape(-1,2)
                warped.append([float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())])
        hints=case['result']['rescue_hints'];coords=lambda p:[p[k] for k in ('left','top','right','bottom')]
        best=[max((box_iou(t,coords(h['box'])) for h in hints),default=0) for t in warped]
        case['evaluation']=dict(target_count=len(targets),target_iou050=sum(v>=.5 for v in best),
                                target_any_overlap=sum(v>0 for v in best),target_best_ious=best)
        kind=name.split('_')[0];g=groups.setdefault(kind,dict(images=0,hint_images=0,precise_images=0,precise_fragments=0,fallbacks=0))
        g['images']+=1;g['hint_images']+=bool(hints);g['precise_images']+=any(v>=.5 for v in best)
        g['precise_fragments']+=sum(v>=.5 for v in best);g['fallbacks']+=case['result']['status']!='applied'
        image_path=OUT/(Path(name).stem+'_aligned.jpg')
        if image_path.exists():
            image=read_image(image_path)
            for t in warped:
                l,top,r,b=map(round,t);cv2.rectangle(image,(l,top),(r,b),(255,230,0),3)
            for h in hints:
                l,top,r,b=map(round,coords(h['box']));cv2.rectangle(image,(l,top),(r,b),(0,165,255),5)
            ok,encoded=cv2.imencode('.jpg',image);assert ok;encoded.tofile(str(OUT/(Path(name).stem+'_overlay.jpg')))
    # Actual SAM reports retain all candidates and the existing local-ECC refusal.
    real=[]
    for case in load(ROOT/'artifacts/main_scenario_live_20261002/report.json')['cases']:
        original=load(Path(case['output'])/'report.json');before=copy.deepcopy(original)
        cached=load(OLD/(Path(case['image']).stem+'_predictions.json'))['predictions']
        result=run_rescue(original,project=REPO,reference_predictions=reference_predictions,roi=roi,enabled=True,
            scene='mendeley_pc_wiring_same_camera',prediction_provider=lambda image,c=cached:copy.deepcopy(c))
        assert original==before and result['parents']==before['review_regions']
        real.append(dict(image=case['image'],status=result['status'],reason=result['fallback_reason'],hints=len(result['rescue_hints'])))
    assert sha(weight)==manifest['weight_sha256'] and all(sha(DATA/'images/test01'/n)==pins[n] for n in pins)
    save(OUT/'report.json',dict(status='complete',groups=groups,cases=cases,real_sam_report_checks=real,
        roi=roi,fresh_reference_port_inference=True,fresh_global_registration=True,
        source_port_predictions='pinned prior inference replay',new_dino_sam_inference=False,
        labels_used_for_selection=False,formal_mainline_changed=False,elapsed_seconds=round(time.perf_counter()-start,2)))
    print('COMPLETE '+json.dumps(groups),flush=True)

if __name__=='__main__':main()
