"""One frozen 784/42x56 CNN train/val experiment; never reads test01."""
import argparse
import copy
import hashlib
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


def nearby_distance(query,normalized_bank,k=3,radius=2):
    query=query/np.maximum(np.linalg.norm(query,axis=-1,keepdims=True),1e-8)
    n,h,w,_=normalized_bank.shape
    best=np.full((n,h,w),-1,dtype=np.float32)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            ya,yb=max(0,-dy),min(h,h-dy);xa,xb=max(0,-dx),min(w,w-dx)
            if ya>=yb or xa>=xb:continue
            value=np.einsum('nhwc,hwc->nhw',normalized_bank[:,ya+dy:yb+dy,xa+dx:xb+dx],query[ya:yb,xa:xb],optimize=True)
            best[:,ya:yb,xa:xb]=np.maximum(best[:,ya:yb,xa:xb],value)
    distance=np.clip(1-best,0,2)
    return np.partition(distance,k-1,axis=0)[:k].mean(axis=0)


def extract(prefix,crop,max_edge,grid,combine):
    scale=max_edge/max(crop.shape[:2]);h,w=max(1,round(crop.shape[0]*scale)),max(1,round(crop.shape[1]*scale))
    resized=cv2.resize(crop,(w,h),interpolation=cv2.INTER_AREA)
    value=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).float()[None]/255
    maps=[]
    with torch.inference_mode():
        for index,layer in enumerate(prefix):
            value=layer(value)
            if index in (2,4):maps.append(value)
        return combine(maps,grid)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('use a new output')
    start=time.perf_counter();args.output.mkdir(parents=True)
    config=args.output/'ultralytics_config';config.mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(config.resolve());os.environ['YOLO_OFFLINE']='True';os.environ['YOLO_AUTOINSTALL']='False'
    sys.path.insert(0,str(args.repo));sys.path.insert(0,str(args.repo/'prototype'))
    from ultralytics import YOLO
    from tools.merge_audit import setup,save
    from tools.prepare_cnn_heat_evidence import combine_maps
    from tools.prepare_fine_heat_evidence import mean_model
    from tools.probe_spatial_metric import unweighted_score
    from tools.probe_local_normal_bank import local_distance,region_score
    from tools.probe_spatial_normal import normal_scale,fuse
    from tools.evaluate_anchored_local import anchored_selection
    from tools.probe_cnn_qualified_support import qualifies
    from tools.audit_cnn_precision_ceiling import reconstruct_groups
    from tools.evaluate_mendeley_local_refinement import summarize,iou,box_area
    from inspection_agent.focus_hint import select_focus_hint
    _,tiled=setup(args.repo);tiled.LARGE_ROI_MAX_CANDIDATES=6
    from evaluate_mendeley_balanced import read_image
    import assembly_auto_review_robust_v3 as perspective
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    snapshot=args.repo/'output/mendeley_cnn_heat_evidence_20260929'
    old=load(snapshot/'original/report.json')
    frozen=load(args.repo/'output/mendeley_cnn_qualified_support_20260929/report.json')
    source_sha=digest(args.repo/'prototype/tiled_dino_review.py')
    assert source_sha==old['source_sha256'] and digest(args.repo/'tools/probe_cnn_qualified_support.py')==frozen['experiment_sha256']
    weights=args.repo/'models/yolov8s-seg.pt';assert weights.is_file() and digest(weights)==old['weights_sha256']
    versions={n:importlib.metadata.version(n) for n in old['versions']};assert versions==old['versions']
    assert len(old['fit_names'])==80 and len(old['embargo_names'])==10 and len(old['calibration_names'])==30
    assert len(set(old['fit_names']+old['embargo_names']+old['calibration_names']))==120
    torch.set_num_threads(4);torch.manual_seed(20260929)
    network=YOLO(str(weights)).model.cpu().eval();prefix=list(network.model[:5])
    assert [type(l).__name__ for l in prefix]==['Conv','Conv','C2f','Conv','C2f'] and all(l.f==-1 for l in prefix)
    manifest_old={(m['split'],m['image']):m for m in old['feature_manifest']}
    dataset=args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images'
    reference=read_image(dataset/'train01/normal_073.JPG');bounds=old['bounds'];x,y,right,bottom=bounds
    assert hashlib.sha256(reference.tobytes()).hexdigest()==manifest_old['train01',old['fit_names'][0]]['identity']['canonical']
    assert digest(args.repo/'prototype/assembly_auto_review_robust_v3.py')==manifest_old['train01',old['fit_names'][0]]['identity']['alignment']
    grid=(42,56);feature_dir=args.output/'features';feature_dir.mkdir()
    manifest=[];timings=[];observed_rss=[]
    proc=psutil.Process();key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    records=[load(p) for directory in ('mendeley_merge_audit_20260928','mendeley_normal_evidence_20260928')
             for p in sorted((args.repo/'output'/directory).glob('*.json')) if p.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' and r['source_sha256']==source_sha for r in records)
    def features(split,name):
        initial=time.perf_counter();path=dataset/split/name
        cv2.setRNGSeed(0);aligned,alignment=perspective.automatic_homography(reference,read_image(path))
        if aligned is None:raise RuntimeError('registration failed: '+name)
        align_seconds=time.perf_counter()-initial;crop=aligned[y:bottom,x:right]
        coarse=extract(prefix,crop,392,(21,28),combine_maps)
        legacy=manifest_old[split,name];cached=snapshot/'features'/legacy['cache']
        assert digest(cached)==legacy['cache_sha256'] and digest(path)==legacy['identity']['source_sha256']
        with np.load(cached,allow_pickle=False) as data:expected=data['features']
        error=float(np.max(np.abs(coarse-expected)))
        np.testing.assert_allclose(coarse,expected,atol=1e-6,rtol=1e-6,err_msg='coarse registration/control drift '+name)
        before=time.perf_counter();fine=extract(prefix,crop,784,grid,combine_maps);fine_seconds=time.perf_counter()-before
        assert fine.shape==(42,56,192) and np.isfinite(fine).all()
        target=feature_dir/(split+'_'+Path(name).stem+'.npz');np.savez_compressed(target,features=fine)
        manifest.append({'split':split,'image':name,'cache':target.name,'cache_sha256':digest(target),
                 'source_sha256':digest(path),'aligned_sha256':hashlib.sha256(aligned.tobytes()).hexdigest(),
                 'alignment':alignment,'coarse_max_abs_error':error})
        timings.append({'split':split,'image':name,'registration_seconds':align_seconds,'fine_extraction_seconds':fine_seconds})
        observed_rss.append(proc.memory_info().rss)
        save(args.output/'progress.json',{'stage':'features','count':len(manifest),'last':name,'manifest':manifest,'timings':timings})
        print(f'features {len(manifest)}/140 {split}/{name} coarse_diff={error:.2g} fine={fine_seconds:.3f}s',flush=True)
        return fine
    bank=np.stack([features('train01',n) for n in old['fit_names']])
    channels=np.array(old['selected_channels']);model=mean_model(bank,channels)
    normalized=bank/np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-8)
    del bank
    local_cal=[];position_cal=[]
    for index,name in enumerate(old['calibration_names']):
        query=features('train01',name);before=time.perf_counter()
        local=nearby_distance(query,normalized)
        if index==0:
            np.testing.assert_allclose(local,local_distance(query,normalized,k=3,radius=2),atol=1e-6,rtol=1e-6)
        local_cal.append(local);position_cal.append(unweighted_score(query,model))
        timings[-1]['score_seconds']=time.perf_counter()-before;observed_rss.append(proc.memory_info().rss)
        print(f'calibration {index+1}/30 score={timings[-1]["score_seconds"]:.2f}s',flush=True)
    local_scale,_=normal_scale(local_cal);position_scale,_=normal_scale(position_cal)
    fused=[fuse(l,p,local_scale,position_scale) for l,p in zip(local_cal,position_cal)]
    threshold=float(np.percentile(np.stack(fused).max(axis=(1,2)),95))
    np.savez_compressed(args.output/'calibration_maps.npz',local=np.stack(local_cal),position=np.stack(position_cal),fusion=np.stack(fused))
    np.savez_compressed(args.output/'mean_model.npz',**model)
    save(args.output/'calibration.json',{'threshold':threshold,'local_scale':local_scale,'position_scale':position_scale,
              'fit_names':old['fit_names'],'embargo_names':old['embargo_names'],'calibration_names':old['calibration_names']})
    print(f'FROZEN train threshold={threshold:.9f}; validation starts',flush=True)
    expected_cases={r['image']:r for r in frozen['results']['cnn_calibrated_support']['cases']}
    cases=[];selection_changes=[]
    for index,r in enumerate(records,1):
        query=features('val01',r['image']);before=time.perf_counter()
        score=fuse(nearby_distance(query,normalized),unweighted_score(query,model),local_scale,position_scale)
        timings[-1]['score_seconds']=time.perf_counter()-before;observed_rss.append(proc.memory_info().rss)
        np.savez_compressed(args.output/(Path(r['image']).stem+'_maps.npz'),fusion=score)
        t=r['trace'];pool=tiled._merge_candidates(copy.deepcopy(t['raw']));tiled._annotate_roi_edges(pool,t['width'],t['height'])
        pool,suppressed=tiled._publish_candidates(pool,[]);assert not suppressed
        qualified=[b for b in pool if qualifies(b,region_score(b,score,t['width'],t['height']),threshold)]
        baseline=t['local_candidates'];selected=anchored_selection(baseline,qualified,score,t['width'],t['height'],region_score)
        assert len(selected)==len(baseline) and (not selected or selected[0]==baseline[0])
        translate=lambda b:{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y}
        parents=copy.deepcopy(expected_cases[r['image']]['candidates'])
        groups=reconstruct_groups(t['raw'],tiled);group_map={key(tiled._merge_candidates(g)[0]):g for g in groups}
        hints=[]
        for parent in parents:
            local={**parent,'left':parent['left']-x,'right':parent['right']-x,'top':parent['top']-y,'bottom':parent['bottom']-y}
            hint=select_focus_hint(local,group_map[key(local)],score,t['width'],t['height'],threshold,expected_grid=grid)
            if hint:hints.append(translate(hint['box']))
        case={'image':r['image'],'targets':r['targets'],'parents':parents,'fine_selected':[translate(b) for b in selected],'hints':hints}
        cases.append(case);selection_changes.append({'image':r['image'],'normal':r['image'].startswith('normal_'),
                    'geometry_changed':set(map(key,parents))!=set(map(key,case['fine_selected']))})
        print(f'validation {index}/30 {r["image"]} hints={len(hints)} score={timings[-1]["score_seconds"]:.2f}s',flush=True)
        save(args.output/'progress.json',{'stage':'validation','count':index,'manifest':manifest,'timings':timings,'cases':cases})
    faults=[r for r in cases if r['targets']];normals=[r for r in cases if not r['targets']]
    results={mode:{'faults':summarize(faults,mode),'normal_regions':sum(len(r[mode]) for r in normals),
                  'by_kind':{k:summarize([r for r in faults if r['image'].startswith(k+'_')],mode) for k in ('damaged','disconnected','misrouted')}}
             for mode in ('parents','fine_selected','hints')}
    assert results['parents']['faults']==frozen['results']['cnn_calibrated_support']['overall']['faults']
    additional={str(cut):sum(max((iou(b,target) for b in c['parents']),default=0)<cut<=max((iou(b,target) for b in c['hints']),default=0)
                for c in faults for target in c['targets']) for cut in (.1,.5)}
    report={'status':'complete','split':'val01','input_max_edge':784,'grid':[42,56,192],'radius':2,'threshold':threshold,
            'results':results,'additional_hint_source_fragments':additional,'selection_changes':selection_changes,'cases':cases,
            'manifest':manifest,'timings':timings,'elapsed_seconds':time.perf_counter()-start,
            'peak_observed_rss_bytes':max(observed_rss),'bank_bytes':normalized.nbytes,
            'weights_sha256':digest(weights),'source_sha256':source_sha,'script_sha256':digest(__file__),
            'coarse_report_sha256':digest(snapshot/'original/report.json'),'versions':versions,'formal_path_changed':False,
            'warning':'Validation-only fixed resolution prototype; hint burden separate. Observed RSS is not guaranteed true peak. Same-chassis fragmented-label metrics are not field accuracy.'}
    save(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('results','additional_hint_source_fragments','elapsed_seconds','peak_observed_rss_bytes')},indent=2),flush=True)


if __name__=='__main__':main()
