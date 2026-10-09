"""One existing generic wire checkpoint, fresh reference only; not RT-DLO."""
from pathlib import Path
import json
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from core import sha256
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_specialized_wire_reference_20261006';out.mkdir(exist_ok=False)
    before=source_pins()
    scope=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json').read_text(encoding='utf-8'))
    approval=json.loads((ROOT/'artifacts/mendeley_reference_confirmed_20261006/approval_record.json').read_text(encoding='utf-8'))
    if before!=approval['mainline_pins']: raise ValueError('mainline drift')
    source=Path(scope['reference_image_path'])
    checkpoint=Path('E:/PythonProject10/output/electric_wires_cpu_poc_20260824/yolov8s_seg_64samples_cpu_30e/weights/best.pt')
    expected='a9362a23a5063d7471146051a1d15e1b31927dc7d5b8e860ac5bf86fc46ee6ef'
    if sha256(checkpoint)!=expected: raise ValueError('checkpoint drift')
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=expected,source=str(source),source_sha256=sha256(source),
        mainline_pins=before,imgsz=640,retrieval_confidence=.25,geometry_readiness_score=.75,
        source_kind='existing64sample generic wire segmentation checkpoint, not new Mendeley training',
        original_crop=[1380,870,1780,1270],fresh_inference=True,retina_masks=True,mask_closing=False,
        reference_only=True,inspection_labels_read=False,code_sha256=sha256(Path(__file__)))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rgb=np.asarray(Image.open(source).convert('RGB'))[870:1270,1380:1780].copy()
    Image.fromarray(rgb).save(out/'fresh_original_crop.png')
    start=time.monotonic(); model=YOLO(str(checkpoint))
    result=model.predict(source=rgb[:,:,::-1].copy(),imgsz=640,conf=.25,device='cpu',
                         retina_masks=True,verbose=False,save=False)[0]
    rows=[]
    if result.masks is not None:
        for i,(native,box) in enumerate(zip(result.masks.data.cpu().numpy(),result.boxes)):
            mask=native>.5
            if mask.shape!=(400,400):raise ValueError('native source-size mask required')
            path=out/f'mask_{i+1:03d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
            n,labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
            components=[]
            for component in range(1,n):
                active=labels==component; hits={}
                for anchor in scope['anchors']:
                    l,t,r,b=anchor['bbox_xyxy'];l-=1380;r-=1380;t-=870;b-=870
                    hits[anchor['id']]=int(active[t:b+1,l:r+1].sum())
                components.append(dict(pixels=int(active.sum()),anchor_hits=hits,
                    spans_two_anchors=all(hits.values())))
            rows.append(dict(mask=str(path),mask_sha256=sha256(path),score=float(box.conf.item()),
                class_id=int(box.cls.item()),class_name=model.names[int(box.cls.item())],components=components))
    ready=any(r['score']>=.75 and any(c['spans_two_anchors'] for c in r['components']) for r in rows)
    Image.fromarray(result.plot()[:,:,::-1]).save(out/'actual_model_overlay.png')
    if before!=source_pins() or sha256(checkpoint)!=expected:raise ValueError('E/input drift')
    report=dict(status='complete',seconds=time.monotonic()-start,native_masks=rows,
        reference_geometry_readiness_passed=ready,reference_only_not_source_gate=True,
        fresh_original_decode=True,fresh_model_inference=True,model_names=model.names,
        mainline_unchanged=True,deployed=False,new_confirmed_connections=0,
        electrical_continuity='not_assessed',not_RT_DLO=True,
        next='fixed four source controls required' if ready else 'reject before source/inspection batch')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(out/'report.json'),seconds=report['seconds'],masks=len(rows),
        scores=[r['score'] for r in rows],reference_ready=ready)))
if __name__=='__main__':main()
