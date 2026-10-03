"""One fixed tiled port experiment with preserved full-frame hints; no test01."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import psutil
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True);(args.output/'ultralytics_config').mkdir();(args.output/'cache').mkdir()
    os.environ['YOLO_CONFIG_DIR']=str((args.output/'ultralytics_config').resolve())
    os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
    sys.path.insert(0,str(args.repo));sys.path.insert(0,str(args.repo/'prototype'))
    from ultralytics import YOLO
    from tools.evaluate_port_state_hints import sha,transform_boxes
    from tools.merge_audit import setup
    setup(args.repo)
    from evaluate_mendeley_balanced import read_image
    from inspection_agent.port_tiling import tile_windows,near_artificial_edge,merge_tiled_ports,select_additional_tile_hint
    from tools.evaluate_mendeley_local_refinement import iou
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    start=time.perf_counter();proc=psutil.Process()
    previous_path=args.repo/'output/port_state_hints_validation_20260929/report.json'
    previous=load(previous_path);assert previous['status']=='complete' and previous['split']=='val01'
    for path,digest in previous['source_hashes'].items():assert sha(path)==digest
    assert all(importlib.metadata.version(n)==v for n,v in previous['versions'].items())
    identities={(m['split'],m['image']):m['source_sha256'] for m in previous['source_manifest']}
    weight=args.repo/'output/mendeley_port_state_cpu_poc_20260825/yolov8s_rectports_3e_960/weights/best.pt'
    fingerprints={str(path):sha(path) for path in (weight,previous_path,Path(__file__),
        args.repo/'inspection_agent/port_tiling.py',args.repo/'inspection_agent/port_state_hint.py',
        args.repo/'tools/evaluate_port_state_hints.py')}
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    reference_path=dataset/'train01/normal_073.JPG';reference=read_image(reference_path)
    fingerprints[str(reference_path)]=sha(reference_path)
    torch.set_num_threads(4);torch.manual_seed(20260929)
    model=YOLO(str(weight));assert model.names=={0:'unplugged_plug',1:'unplugged_jack'}
    manifests=[];timings=[];calibration=[];cases=[]
    def predict(split,name):
        source=dataset/split/name;digest=sha(source);assert digest==identities[split,name]
        image=read_image(source);h,w=image.shape[:2];windows=tile_windows(w,h)
        kept=[];raw_count=0;edge_count=0;tick=time.perf_counter();sampled_rss=[]
        for offset in range(0,len(windows),2):
            group=windows[offset:offset+2]
            crops=[image[t:b,l:r] for l,t,r,b in group]
            outputs=model.predict(crops,imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)
            assert len(outputs)==len(group)
            for number,(result,window) in enumerate(zip(outputs,group),offset):
                x,y,r,b=window
                for box in result.boxes:
                    raw_count+=1;local=list(map(float,box.xyxy[0].tolist()))
                    if near_artificial_edge(local,window,w,h):edge_count+=1;continue
                    l,t,rr,bb=local
                    kept.append({'box_xyxy':[l+x,t+y,rr+x,bb+y],
                        'confidence':float(box.conf.item()),'class_id':int(box.cls.item()),'source_tile':number})
            sampled_rss.append(proc.memory_info().rss/2**30)
        merged=merge_tiled_ports(kept);seconds=time.perf_counter()-tick
        assert sha(source)==digest
        cache_name=split+'_'+Path(name).stem+'.json'
        record={'image':name,'split':split,'shape':[h,w],'source_sha256':digest,
            'windows':[list(window) for window in windows],'raw_prediction_count':raw_count,
            'edge_rejected_count':edge_count,'edge_kept_predictions':kept,'merged_predictions':merged}
        save(args.output/'cache'/cache_name,record)
        manifests.append({'image':name,'split':split,'source_sha256':digest,'cache':cache_name,
            'cache_sha256':sha(args.output/'cache'/cache_name)})
        timings.append({'image':name,'split':split,'tile_count':len(windows),
            'seconds':seconds,'sampled_rss_GiB':max(sampled_rss)})
        return image,merged
    for index,item in enumerate(previous['calibration'],1):
        image,merged=predict('train01',item['image'])
        calibration.append({'image':item['image'],'maximum_confidence':max((r['confidence'] for r in merged),default=0.)})
        save(args.output/'progress.json',{'phase':'train_normal','completed':index,'total':30})
        print('train normal '+str(index)+'/30 | merged '+str(len(merged)),flush=True)
    assert len(calibration)==30
    threshold=max(.25,float(np.quantile([c['maximum_confidence'] for c in calibration],.95)))
    save(args.output/'calibration.json',{'threshold':threshold,'samples':calibration,
        'warning':'Train-seen normals and prior validation-selected model; not independent calibration.'})
    print('Frozen tile threshold '+str(threshold),flush=True)
    for index,old in enumerate(sorted(previous['cases'],key=lambda c:c['image']),1):
        image,merged=predict('val01',old['image'])
        matrix=np.asarray(old['actual_homography']);assert old['alignment']['alignment_quality']['reliable']
        valid=cv2.warpPerspective(np.full(image.shape[:2],255,np.uint8),matrix,
            (reference.shape[1],reference.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        valid=cv2.erode(valid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
        transformed=[]
        for raw,bounds in zip(merged,transform_boxes([r['box_xyxy'] for r in merged],matrix)):
            l,t,r,b=bounds;crop=valid[max(0,t):min(valid.shape[0],b),max(0,l):min(valid.shape[1],r)]
            coverage=float(crop.sum()/max(1,(r-l)*(b-t))) if crop.size else 0.
            transformed.append({**dict(zip(('left','top','right','bottom'),bounds)),
                'class_id':raw['class_id'],'confidence':raw['confidence'],'valid_warp_fraction':coverage,
                'support_tiles':raw['support_tiles']})
        selection=select_additional_tile_hint(old['parents'],old['hints'],transformed,threshold)
        assert selection['parents']==old['parents'] and selection['existing_hints']==old['hints']
        # Cached annotations are accessed for scoring only after selection.
        cases.append({'image':old['image'],'kind':old['kind'],'split':'val01',
            'targets':copy.deepcopy(old['targets']),'source_classes':copy.deepcopy(old['source_classes']),
            'actual_homography':old['actual_homography'],'aligned_predictions':transformed,**selection})
        save(args.output/'partial_cases.json',cases)
        save(args.output/'progress.json',{'phase':'validation','completed':index,'total':30})
        print('validation '+str(index)+'/30 | '+old['image']+' | new hints '+str(len(selection['tile_hints'])),flush=True)
    def boxes(case,added):
        return case['parents']+[h['box'] for h in case['existing_hints']]+([h['box'] for h in case['tile_hints']] if added else [])
    def summary(added):
        faults=[c for c in cases if c['targets']];normals=[c for c in cases if not c['targets']]
        per=lambda c:[max((iou(b,t) for b in boxes(c,added)),default=0.) for t in c['targets']]
        values=[v for c in faults for v in per(c)]
        return {'fault_images_with_overlap':sum(any(v>0 for v in per(c)) for c in faults),
            'source_fragments':len(values),'overlap_fragments':sum(v>0 for v in values),
            'iou_ge_01':sum(v>=.1 for v in values),'iou_ge_05':sum(v>=.5 for v in values),
            'mean_best_iou':float(np.mean(values)),'fault_regions':sum(len(boxes(c,added)) for c in faults),
            'normal_regions':sum(len(boxes(c,added)) for c in normals),
            'normal_images_with_added_hints':sum(bool(c['tile_hints']) for c in normals)}
    baseline=summary(False);tiled=summary(True)
    assert baseline['iou_ge_05']==5 and baseline['overlap_fragments']==164
    assert baseline['fault_regions']==71 and baseline['normal_regions']==37
    per_image=[];tp=fp=fn=0
    for case in cases:
        old=[max((iou(b,t) for b in boxes(case,False)),default=0.) for t in case['targets']]
        new=[max((iou(b,t) for b in boxes(case,True)),default=0.) for t in case['targets']]
        per_image.append({'image':case['image'],'new_hint_count':len(case['tile_hints']),
            'extra_precise':sum(n>=.5 and o<.5 for o,n in zip(old,new)),
            'extra_overlap':sum(n>0 and o==0 for o,n in zip(old,new))})
        hints=[h['box'] for h in case['tile_hints']]
        targets=[(t,cls) for t,cls in zip(case['targets'],case['source_classes']) if cls in (3,4)]
        possibilities=sorted([(iou(b,t),bi,ti) for bi,b in enumerate(hints) for ti,(t,cls) in enumerate(targets)
                              if b['class_id']+3==cls],reverse=True)
        used_b=set();used_t=set()
        for overlap,bi,ti in possibilities:
            if overlap>=.5 and bi not in used_b and ti not in used_t:used_b.add(bi);used_t.add(ti)
        tp+=len(used_b);fp+=len(hints)-len(used_b);fn+=len(targets)-len(used_t)
    result={'status':'complete','split':'val01','threshold':threshold,'tile_size':1280,'stride':960,
        'edge_margin':16,'cross_tile_nms_iou':.5,'maximum_new_hints_per_image':1,
        'calibration':calibration,'baseline':baseline,'with_tiled_hints':tiled,'per_image':per_image,'cases':cases,
        'new_hint_class_matched_metrics_at_iou_05':{'tp':tp,'unmatched_hints':fp,'fn':fn},
        'source_manifest':manifests,'fingerprints':fingerprints,'versions':previous['versions'],'timings':timings,
        'elapsed_seconds':time.perf_counter()-start,'model_tile_passes':sum(t['tile_count'] for t in timings),
        'sampled_max_rss_GiB':max(t['sampled_rss_GiB'] for t in timings),'formal_detection_changed':False,
        'warning':'Fixed retrospective validation only. Old Dell model, train-seen normal calibration, prior val weight selection. No field accuracy, cable identity, electrical continuity or cross-scene generalization claim.'}
    for path,digest in fingerprints.items():assert sha(path)==digest
    save(args.output/'report.json',result);save(args.output/'progress.json',{'phase':'complete','completed':60,'total':60})
    print(json.dumps({k:v for k,v in result.items() if k in ('threshold','baseline','with_tiled_hints',
        'new_hint_class_matched_metrics_at_iou_05','elapsed_seconds','model_tile_passes','sampled_max_rss_GiB')},indent=2),flush=True)


if __name__=='__main__':main()
