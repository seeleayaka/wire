"""Frozen retrospective train-normal/val port cue audit; never reads test01."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def transform_boxes(boxes, matrix):
    result=[]
    for l,t,r,b in boxes:
        corners=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
        result.append([int(np.floor(corners[:,0].min())),int(np.floor(corners[:,1].min())),
                       int(np.ceil(corners[:,0].max())),int(np.ceil(corners[:,1].max()))])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('use fresh output')
    args.output.mkdir(parents=True); started=time.perf_counter()
    (args.output/'ultralytics_config').mkdir()
    os.environ['YOLO_CONFIG_DIR']=str((args.output/'ultralytics_config').resolve())
    os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
    sys.path.insert(0,str(args.repo));sys.path.insert(0,str(args.repo/'prototype'))
    from ultralytics import YOLO
    from tools.merge_audit import setup
    setup(args.repo)
    from evaluate_mendeley_balanced import read_image,load_target_boxes
    import assembly_auto_review_robust_v3 as perspective
    from inspection_agent.port_state_hint import select_port_state_hints
    from tools.evaluate_mendeley_local_refinement import iou
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    save=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
    frozen=load(args.repo/'output/mendeley_cnn_qualified_support_20260929/report.json')
    legacy=load(args.repo/'output/mendeley_cnn_heat_evidence_20260929/original/report.json')
    identities={(m['split'],m['image']):m['identity'] for m in legacy['feature_manifest']}
    weight=args.repo/'output/mendeley_port_state_cpu_poc_20260825/yolov8s_rectports_3e_960/weights/best.pt'
    source_hashes={str(path):sha(path) for path in (weight,weight.parents[1]/'args.yaml',Path(__file__),
                   args.repo/'inspection_agent/port_state_hint.py',args.repo/'prototype/tiled_dino_review.py',
                   args.repo/'prototype/assembly_auto_review_robust_v3.py')}
    assert sha(args.repo/'prototype/tiled_dino_review.py')==legacy['source_sha256']
    assert sha(args.repo/'prototype/assembly_auto_review_robust_v3.py')==identities['train01',legacy['fit_names'][0]]['alignment']
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    torch.set_num_threads(4);torch.manual_seed(20260929)
    model=YOLO(str(weight));assert model.task=='segment'
    assert dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    manifest=[];calibration=[];cases=[];timings=[]
    def predict(split,name):
        path=dataset/'images'/split/name
        fingerprint=sha(path);assert fingerprint==identities[split,name]['source_sha256']
        image=read_image(path);tick=time.perf_counter()
        result=model.predict(image,imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)[0]
        predictions=[{'class_id':int(box.cls.item()),'confidence':float(box.conf.item()),
                      'box_xyxy':list(map(float,box.xyxy[0].tolist()))} for box in result.boxes]
        assert all(r['class_id'] in (0,1) for r in predictions)
        assert fingerprint==sha(path)
        seconds=time.perf_counter()-tick
        timings.append({'split':split,'image':name,'inference_seconds':seconds})
        manifest.append({'split':split,'image':name,'source_sha256':fingerprint})
        return image,predictions
    for index,name in enumerate(legacy['calibration_names'],1):
        assert name.startswith('normal_') and name.endswith('.JPG')
        _,raw=predict('train01',name)
        calibration.append({'image':name,'maximum_confidence':max((r['confidence'] for r in raw),default=0.),'predictions':raw})
        print('normal calibration '+str(index)+'/30',flush=True)
    assert len(calibration)==30
    threshold=max(.25,float(np.quantile([r['maximum_confidence'] for r in calibration],.95)))
    save(args.output/'calibration.json',{'threshold':threshold,'samples':calibration,
         'warning':'These normal samples participated in old detector training; not independent detector calibration.'})
    parents_by_name={c['image']:c['candidates'] for c in frozen['results']['cnn_calibrated_support']['cases']}
    assert len(parents_by_name)==30
    for index,name in enumerate(sorted(parents_by_name),1):
        image,raw=predict('val01',name)
        captured=[];original=cv2.findHomography
        def capture(*a,**kw):
            value=original(*a,**kw);captured.append(value[0]);return value
        cv2.findHomography=capture;tick=time.perf_counter()
        try:
            cv2.setRNGSeed(0);aligned,alignment=perspective.automatic_homography(reference,image)
        finally:cv2.findHomography=original
        assert aligned is not None and len(captured)==1 and captured[0] is not None
        matrix=captured[0];valid=perspective.auto.LAST_WARP_VALID_MASK>0
        timings[-1]['alignment_seconds']=time.perf_counter()-tick
        transformed=[]
        for source_record,bounds in zip(raw,transform_boxes([r['box_xyxy'] for r in raw],matrix)):
            l,t,r,b=bounds
            crop=valid[max(0,t):min(valid.shape[0],b),max(0,l):min(valid.shape[1],r)]
            coverage=float(crop.sum()/max(1,(r-l)*(b-t))) if crop.size else 0.
            transformed.append({**dict(zip(('left','top','right','bottom'),bounds)),
                'class_id':source_record['class_id'],'confidence':source_record['confidence'],
                'valid_warp_fraction':coverage})
        selected=select_port_state_hints(parents_by_name[name],transformed,threshold)
        assert selected['parents']==parents_by_name[name]
        # Labels enter only after predictions and hint selection have completed.
        originals=load_target_boxes(dataset/'labels/val01'/(Path(name).stem+'.txt'),image.shape[1],image.shape[0])
        targets=transform_boxes(originals,matrix)
        label_classes=[int(line.split()[0]) for line in (dataset/'labels/val01'/(Path(name).stem+'.txt')).read_text().splitlines() if line.strip()]
        trace_dir='mendeley_normal_evidence_20260928' if name.startswith('normal_') else 'mendeley_merge_audit_20260928'
        trace=load(args.repo/'output'/trace_dir/(Path(name).stem+'.json'))
        assert targets==trace['targets'],name+' scoring coordinates changed'
        case={'image':name,'kind':name.split('_')[0],'split':'val01','targets':targets,'source_classes':label_classes,
              'source_predictions':raw,'aligned_predictions':transformed,'alignment':alignment,
              'actual_homography':matrix.tolist(),**selected}
        cases.append(case);save(args.output/'partial_cases.json',cases)
        print('validation '+str(index)+'/30 | '+name+' | hints '+str(len(case['hints'])),flush=True)
        if name==min(n for n in parents_by_name if n.startswith(case['kind']+'_')):
            panel=Image.fromarray(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(panel)
            for box in case['parents']:draw.rectangle([box[k] for k in ('left','top','right','bottom')],outline='yellow',width=4)
            for hint in case['hints']:draw.rectangle([hint['box'][k] for k in ('left','top','right','bottom')],outline='cyan',width=6)
            for target in targets:draw.rectangle(target,outline='red',width=2)
            panel.thumbnail((950,715));panel.save(args.output/(Path(name).stem+'_hints.png'))
    def summary(mode):
        faults=[c for c in cases if c['targets']];normals=[c for c in cases if not c['targets']]
        boxes=lambda c:c['parents']+([h['box'] for h in c['hints']] if mode=='with_hints' else [])
        best=lambda c:[max((iou(b,t) for b in boxes(c)),default=0.) for t in c['targets']]
        values=[v for c in faults for v in best(c)]
        return {'fault_images':len(faults),'fault_images_with_overlap':sum(any(v>0 for v in best(c)) for c in faults),
                'source_fragments':len(values),'overlap_fragments':sum(v>0 for v in values),
                'iou_ge_01':sum(v>=.1 for v in values),'iou_ge_05':sum(v>=.5 for v in values),
                'mean_best_iou':float(np.mean(values)),
                'fault_regions':sum(len(boxes(c)) for c in faults),'normal_regions':sum(len(boxes(c)) for c in normals),
                'normal_images_with_hints':sum(bool(c['hints']) for c in normals)}
    baseline,combined=summary('parents'),summary('with_hints')
    assert baseline['overlap_fragments']==164 and baseline['iou_ge_05']==4
    assert baseline['fault_regions']==70 and baseline['normal_regions']==37
    per_image=[];strict_tp=strict_fp=strict_fn=0
    for c in cases:
        hints=[h['box'] for h in c['hints']];old=[max((iou(b,t) for b in c['parents']),default=0.) for t in c['targets']]
        new=[max([v]+[iou(b,t) for b in hints]) for v,t in zip(old,c['targets'])]
        per_image.append({'image':c['image'],'hint_count':len(hints),'extra_precise':sum(n>=.5 and o<.5 for o,n in zip(old,new)),
                          'extra_overlap':sum(n>0 and o==0 for o,n in zip(old,new))})
        eligible=[i for i,cls in enumerate(c['source_classes']) if cls in (3,4)]
        possible=sorted([(iou(b,c['targets'][i]),bi,i) for bi,b in enumerate(hints) for i in eligible
                         if b['class_id']+3==c['source_classes'][i]],reverse=True)
        used_h=set();used_t=set()
        for overlap,bi,i in possible:
            if overlap>=.5 and bi not in used_h and i not in used_t:used_h.add(bi);used_t.add(i)
        strict_tp+=len(used_h);strict_fp+=len(hints)-len(used_h);strict_fn+=len(eligible)-len(used_t)
    result={'status':'complete','split':'val01','threshold':threshold,'calibration':calibration,
            'baseline':baseline,'with_hints':combined,'per_image':per_image,'cases':cases,
            'class_matched_hint_metrics_at_iou_05':{'tp':strict_tp,'unmatched_hints':strict_fp,'fn':strict_fn,
                'precision':strict_tp/(strict_tp+strict_fp) if strict_tp+strict_fp else None},
            'source_hashes':source_hashes,'source_manifest':manifest,'model_names':model.names,
            'versions':{n:importlib.metadata.version(n) for n in ('torch','ultralytics','numpy','opencv-python')},
            'timings':timings,'elapsed_seconds':time.perf_counter()-started,'formal_detection_changed':False,
            'warning':'Retrospective fixed-Dell source validation, previously validation-selected supervised weights and train-seen normal calibration. No independent field/generalization accuracy, automatic fault verdict or seating verification.'}
    assert sha(weight)==source_hashes[str(weight)]
    save(args.output/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ('threshold','baseline','with_hints','class_matched_hint_metrics_at_iou_05','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
