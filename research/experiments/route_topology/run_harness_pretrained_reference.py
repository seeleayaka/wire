"""Separate nine-class pretrained checkpoint route; fresh reference only."""
import json
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_harness_pretrained_reference_v2_20261006';out.mkdir(exist_ok=False)
    before=source_pins()
    checkpoint=Path('E:/PythonProject10/models/wire_harness_yolov8s_seg_best.pt')
    manifest=json.loads((ROOT/'publication/wire-20261004/backup/model-manifest.json').read_text(encoding='utf-8'))
    record=next(r for r in manifest['model_archive']['entries'] if r['path']=='models/wire_harness_yolov8s_seg_best.pt')
    if sha256(checkpoint)!=record['sha256']: raise ValueError('checkpoint drift')
    scope=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json').read_text(encoding='utf-8'))
    approval=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/approval_record.json').read_text(encoding='utf-8'))
    if before!=approval['mainline_pins']:raise ValueError('mainline drift')
    source=Path(scope['reference_image_path']);rgb=np.asarray(Image.open(source).convert('RGB'))[870:1270,1380:1780].copy()
    (out/'protocol.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),source=str(source),source_sha256=sha256(source),crop_xyxy=[1380,870,1780,1270],confidence=.25,imgsz=640,retina_masks=True,reference_only=True,code_sha256=sha256(Path(__file__))),indent=2),encoding='utf-8')
    Image.fromarray(rgb).save(out/'fresh_original_crop.png')
    start=time.monotonic();model=YOLO(str(checkpoint))
    result=model.predict(rgb[:,:,::-1].copy(),imgsz=640,conf=.25,device='cpu',retina_masks=True,save=False,verbose=False)[0]
    rows=[]
    for i,box in enumerate(result.boxes):
        name=model.names[int(box.cls.item())];score=float(box.conf.item());components=[]
        if result.masks is not None:
            mask=result.masks.data[i].cpu().numpy()>.5
            if mask.shape!=(400,400): raise ValueError('wrong native frame')
            path=out/f'mask_{i+1:03d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
            n,labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
            for k in range(1,n):
                active=labels==k;hits={}
                for a in scope['anchors']:
                    l,t,r,b=a['bbox_xyxy'];hits[a['id']]=int(active[t-870:b-870+1,l-1380:r-1380+1].sum())
                components.append(dict(pixels=int(active.sum()),anchor_hits=hits,spans_two_anchors=all(hits.values())))
        rows.append(dict(class_name=name,score=score,components=components,
            eligible_wire_semantics='wire' in name.lower() or 'cable' in name.lower()))
    ready=any(r['eligible_wire_semantics'] and r['score']>=.75 and any(c['spans_two_anchors'] for c in r['components']) for r in rows)
    Image.fromarray(result.plot()[:,:,::-1]).save(out/'actual_model_overlay.png')
    if before!=source_pins() or sha256(checkpoint)!=record['sha256']:raise ValueError('E drift')
    report=dict(status='complete',seconds=time.monotonic()-start,model_names=model.names,predictions=rows,
        reference_geometry_readiness_passed=ready,fresh_original_decode_and_inference=True,
        mainline_unchanged=True,new_confirmed_connections=0,deployed=False,
        next='fixed sources required' if ready else 'reject reference prerequisite, no source/inspection run')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(reference_ready=ready,seconds=report['seconds'],names=model.names,predictions=len(rows))))
if __name__=='__main__':main()
