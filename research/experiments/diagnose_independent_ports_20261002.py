"""Frozen full-test01 port diagnostic, not a change to the parent-gated GUI policy."""
import sys
import os
import json
import time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/port_bridge_fix_20261002/independent_test01'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path.insert(0,str(REPO))
os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False')
import torch
import cv2
from inspection_agent.optional_port_crop_review import CONFIG,CALIBRATION_SHA,sha,read_image,predict
from inspection_agent.port_tiling import box_iou

def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if OUT.exists():raise FileExistsError('Do not overwrite the diagnostic')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    calibration=REPO/'config/port_crop_calibration_frozen_20260930.json'
    frozen=json.loads(calibration.read_text(encoding='utf-8'))
    weight=REPO/'output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt'
    assert sha(calibration)==CALIBRATION_SHA and CONFIG==frozen['config']
    assert sha(weight)==frozen['fingerprints']['weight']
    paths=sorted((DATA/'images/test01').glob('*.JPG'))
    assert len(paths)==30 and sum(p.name.startswith('normal_') for p in paths)==15
    manifest=dict(images=[dict(name=p.name,sha256=sha(p)) for p in paths],weight_sha256=sha(weight),
        calibration_sha256=sha(calibration),threshold=.25,top_budget=1,
        selection='highest confidence then smallest area then coordinates; no label access',
        independent_of_parent=True,formal_policy_changed=False,already_used_test_set=True)
    save(OUT/'manifest.json',manifest)
    from ultralytics import YOLO
    torch.set_num_threads(4);torch.manual_seed(20260929)
    model=YOLO(str(weight));assert model.task=='segment'
    assert dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    cases=[];start=time.perf_counter()
    for path in paths:
        predictions=predict(model,read_image(path))
        eligible=[p for p in predictions['merged_predictions'] if p['confidence']>.25]
        eligible.sort(key=lambda p:(-p['confidence'],(p['box_xyxy'][2]-p['box_xyxy'][0])*(p['box_xyxy'][3]-p['box_xyxy'][1]),*p['box_xyxy'],p['class_id']))
        case=dict(image=path.name,predictions=predictions,eligible=eligible,top1=eligible[:1])
        cases.append(case);save(OUT/(path.stem+'_predictions.json'),case)
        print(f'PREDICTED {len(cases)}/30 {path.name} strong={len(eligible)}',flush=True)
    # Read labels only after all inference/selection is complete.
    aggregate={}
    for case,path in zip(cases,paths):
        image=read_image(path);height,width=image.shape[:2];targets=[]
        for line in (DATA/'labels/test01'/(path.stem+'.txt')).read_text(encoding='utf-8').splitlines():
            _,cx,cy,w,h=map(float,line.split())
            targets.append([(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height])
        metrics={}
        for stage in ('eligible','top1'):
            boxes=[p['box_xyxy'] for p in case[stage]]
            best=[max((box_iou(t,b) for b in boxes),default=0.) for t in targets]
            metrics[stage]=dict(candidates=len(boxes),target_any_overlap=sum(v>0 for v in best),
                               target_iou050=sum(v>=.5 for v in best),target_best_ious=best)
        case['target_count']=len(targets);case['metrics']=metrics
        kind=path.name.split('_')[0]
        group=aggregate.setdefault(kind,dict(images=0,targets=0,images_with_top1=0,top1_precise_images=0,
            top1_any_overlap=0,top1_iou050=0,all_strong_count=0,all_strong_iou050=0))
        group['images']+=1;group['targets']+=len(targets);group['images_with_top1']+=bool(case['top1'])
        group['top1_precise_images']+=metrics['top1']['target_iou050']>0
        group['top1_any_overlap']+=metrics['top1']['target_any_overlap']
        group['top1_iou050']+=metrics['top1']['target_iou050']
        group['all_strong_count']+=metrics['eligible']['candidates']
        group['all_strong_iou050']+=metrics['eligible']['target_iou050']
        for idx,t in enumerate(targets,1):
            l,top,r,b=map(round,t);cv2.rectangle(image,(l,top),(r,b),(255,230,0),3)
            cv2.putText(image,f'T{idx}',(l,max(28,top)),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,230,0),2)
        for p in case['top1']:
            l,top,r,b=map(round,p['box_xyxy']);cv2.rectangle(image,(l,top),(r,b),(0,165,255),5)
            cv2.putText(image,f"P {p['confidence']:.2f}",(l,max(28,top-5)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,165,255),2)
        ok,encoded=cv2.imencode('.jpg',image);assert ok;encoded.tofile(str(OUT/(path.stem+'_overlay.jpg')))
    assert manifest['images']==[dict(name=p.name,sha256=sha(p)) for p in paths]
    assert sha(weight)==manifest['weight_sha256'] and sha(calibration)==manifest['calibration_sha256']
    save(OUT/'report.json',dict(status='complete',manifest=manifest,cases=cases,aggregate=aggregate,
        elapsed_seconds=round(time.perf_counter()-start,2),new_port_inference=True,
        new_dino_sam_inference=False,human_confirmation=False,field_accuracy_claimed=False))
    print('COMPLETE '+json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
