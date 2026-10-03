"""Single frozen test01 acceptance; all writes stay in the authorized workspace."""
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
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    authorized=Path(__file__).resolve().parent
    output=a.output.resolve()
    if not output.is_relative_to(authorized): raise ValueError('output must stay in this authorized experiments directory')
    if output.exists(): raise FileExistsError('use a fresh output directory')
    output.mkdir(parents=True)
    config=output/'ultralytics_config'; config.mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(config)
    os.environ['YOLO_OFFLINE']='True'; os.environ['YOLO_AUTOINSTALL']='False'
    os.environ['PYTHONPROJECT10_DINO_CACHE']=str(output/'dino_reference_cache')
    sys.path.insert(0,str(a.repo)); sys.path.insert(0,str(a.repo/'prototype'))
    from ultralytics import YOLO
    from tools.merge_audit import setup,save
    from tools.prepare_cnn_heat_evidence import combine_maps
    from tools.probe_local_normal_bank import local_distance,region_score
    from tools.probe_spatial_metric import unweighted_score
    from tools.probe_spatial_normal import fuse
    from tools.probe_cnn_qualified_support import qualifies
    from tools.evaluate_anchored_local import anchored_selection
    from tools.evaluate_mendeley_local_refinement import summarize
    from tools.probe_cnn_candidate_rank import box_union_area
    from tools.evaluate_mendeley_holdout_current import transform_boxes
    impl,tiled=setup(a.repo)
    from evaluate_mendeley_balanced import read_image,load_target_boxes,FULL_REVIEW_ROI
    import assembly_auto_review_robust_v3 as perspective
    evidence=a.repo/'output/mendeley_cnn_heat_evidence_20260929'
    source=json.loads((evidence/'original/report.json').read_text(encoding='utf-8'))
    metric=json.loads((evidence/'metric/report.json').read_text(encoding='utf-8'))
    frozen=json.loads((a.repo/'output/mendeley_cnn_qualified_support_20260929/report.json').read_text(encoding='utf-8'))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen['experiment_sha256']==digest(a.repo/'tools/probe_cnn_qualified_support.py')
    assert source['extractor_sha256']==digest(a.repo/'tools/prepare_cnn_heat_evidence.py')
    assert source['source_sha256']==digest(a.repo/'prototype/tiled_dino_review.py')
    assert source['versions']=={name:importlib.metadata.version(name) for name in source['versions']}
    assert frozen['evidence_report_sha256']==digest(evidence/'original/report.json')
    assert frozen['calibration_maps_sha256']==digest(evidence/'original/calibration_maps.npz')
    assert metric['source_report_sha256']==digest(evidence/'original/report.json')
    manifest={(m['split'],m['image']):m for m in source['feature_manifest']}
    assert digest(a.repo/'prototype/assembly_auto_review_robust_v3.py')==manifest['train01',source['fit_names'][0]]['identity']['alignment']
    fit=[]
    for name in source['fit_names']:
        entry=manifest['train01',name]; path=evidence/'features'/entry['cache']
        assert digest(path)==entry['cache_sha256']
        with np.load(path,allow_pickle=False) as data: fit.append(data['features'])
    bank=np.stack(fit); del fit
    assert bank.shape==(80,21,28,192)
    with np.load(evidence/'original/spatial_model.npz',allow_pickle=False) as data:
        position_model={key:data[key] for key in ('channels','mean')}
    with np.load(evidence/'original/calibration_maps.npz',allow_pickle=False) as data:
        assert float(np.percentile(data['fusion'].max(axis=(1,2)),95))==frozen['threshold']
    weights=a.repo/'models/yolov8s-seg.pt'
    assert digest(weights)==source['weights_sha256']
    torch.set_num_threads(4); torch.manual_seed(20260929)
    network=YOLO(str(weights)).model.cpu().eval(); prefix=list(network.model[:5])
    assert all(layer.f==-1 for layer in prefix)
    assert [type(layer).__name__ for layer in prefix]==['Conv','Conv','C2f','Conv','C2f']
    dataset=a.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    assert hashlib.sha256(reference.tobytes()).hexdigest()==manifest['train01',source['fit_names'][0]]['identity']['canonical']
    bounds=source['bounds']; x,y,right,bottom=bounds
    assert list(impl.adaptive.local_review.motion.perspective.auto.base.pixels(reference,FULL_REVIEW_ROI[0]))==bounds
    paths=sorted((dataset/'images/test01').glob('*.JPG'))
    assert len(paths)==30 and sum(not path.name.startswith('normal_') for path in paths)==15
    historical=json.loads((a.repo/'output/mendeley_reference_localization_20260927/test_fixed_faults_budget6.json').read_text(encoding='utf-8'))
    historical={r['image']:r for r in historical['cases']}
    tiled.LARGE_ROI_MAX_CANDIDATES=6
    original_merge,original_review=tiled._merge_candidates,impl.review_components
    raw=[]; traces=[]
    def merge_hook(rows):
        raw.append(copy.deepcopy(rows)); return original_merge(rows)
    def review_hook(ref,aligned,valid,**kwargs):
        before=len(raw); result=original_review(ref,aligned,valid,**kwargs)
        assert len(raw)==before+1
        traces.append({'raw':raw[-1],'height':ref.shape[0],'width':ref.shape[1],
                       'local_candidates':copy.deepcopy(result[2]),'metadata':copy.deepcopy(result[1])})
        return result
    tiled._merge_candidates,impl.review_components=merge_hook,review_hook
    cases={m:[] for m in ('baseline','cnn_original_eligible','cnn_calibrated_support')}
    audits=[]; started=time.perf_counter()
    try:
        for index,path in enumerate(paths,1):
            image_started=time.perf_counter(); raw.clear(); traces.clear()
            inspection=read_image(path); captured=[]; original_find=cv2.findHomography
            def find_hook(*args,**kwargs):
                result=original_find(*args,**kwargs); captured.append(result[0]); return result
            cv2.setRNGSeed(0)
            try:
                cv2.findHomography=find_hook
                aligned,alignment=perspective.automatic_homography(reference,inspection)
            finally: cv2.findHomography=original_find
            assert aligned is not None and len(captured)==1 and captured[0] is not None
            aligned_hash=hashlib.sha256(aligned.tobytes()).hexdigest()
            cnn_started=time.perf_counter()
            crop=aligned[y:bottom,x:right]; scale=392/max(crop.shape[:2])
            h,w=max(1,round(crop.shape[0]*scale)),max(1,round(crop.shape[1]*scale))
            resized=cv2.resize(crop,(w,h),interpolation=cv2.INTER_AREA)
            tensor=torch.from_numpy(np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1))).float()[None]/255
            feature_maps=[]
            with torch.inference_mode():
                for layer_index,layer in enumerate(prefix):
                    tensor=layer(tensor)
                    if layer_index in (2,4): feature_maps.append(tensor)
                query=combine_maps(feature_maps,(21,28))
            score=fuse(local_distance(query,bank,k=3,radius=1),unweighted_score(query,position_model),
                       source['calibration']['local_scale'],metric['calibration']['identity_scale'])
            cnn_seconds=time.perf_counter()-cnn_started
            dino_started=time.perf_counter()
            _,_,published_absolute=impl.dino_fused_regions(reference,aligned,FULL_REVIEW_ROI)
            dino_seconds=time.perf_counter()-dino_started
            assert len(traces)==1, 'unexpected DINO trace or fallback'
            trace=traces[0]
            assert not trace['metadata']['repetitive_group_candidate_count']
            pool=original_merge(copy.deepcopy(trace['raw']))
            tiled._annotate_roi_edges(pool,trace['width'],trace['height'])
            published,suppressed=tiled._publish_candidates(pool,[]); assert not suppressed
            eligible,_=tiled._display_candidates_for_large_roi(published)
            budget,_=tiled._large_roi_candidate_budget(eligible)
            baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
            assert baseline==trace['local_candidates']
            coords=lambda rows:[[b[k] for k in ('left','top','right','bottom')] for b in rows]
            selected={'baseline':baseline,'cnn_original_eligible':anchored_selection(baseline,eligible,score,trace['width'],trace['height'],region_score)}
            qualified=[b for b in published if qualifies(b,region_score(b,score,trace['width'],trace['height']),frozen['threshold'])]
            selected['cnn_calibrated_support']=anchored_selection(baseline,qualified,score,trace['width'],trace['height'],region_score)
            # Labels and historical fault candidates only enter after inference.
            targets=transform_boxes(load_target_boxes(dataset/'labels/test01'/(path.stem+'.txt'),inspection.shape[1],inspection.shape[0]),captured[0])
            historical_exact=True
            if path.name in historical:
                historical_exact=coords(published_absolute)==coords(historical[path.name]['candidates'])
                assert historical_exact, f'Historical DINO baseline mismatch: {path.name}'
                assert targets==historical[path.name]['target_boxes_aligned_xyxy'], f'Historical label transform mismatch: {path.name}'
            eligible_keys={tuple(b[k] for k in ('left','top','right','bottom')) for b in eligible}
            for mode,rows in selected.items():
                assert len(rows)==len(baseline) and (not rows or rows[0]==baseline[0])
                cases[mode].append({'image':path.name,'targets':targets,'candidates':[
                    {**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in rows]})
            audit={'image':path.name,'split':'test01','bounds':bounds,'alignment':alignment,'actual_homography':captured[0].tolist(),
                   'source_sha256':digest(path),'aligned_sha256':aligned_hash,'trace':trace,'historical_baseline_exact':historical_exact,
                   'cnn_seconds':cnn_seconds,'dino_seconds':dino_seconds,'total_seconds':time.perf_counter()-image_started,
                   'added_qualified_count':sum(tuple(b[k] for k in ('left','top','right','bottom')) not in eligible_keys for b in qualified),
                   'added_selected_count':sum(tuple(b[k] for k in ('left','top','right','bottom')) not in eligible_keys for b in selected['cnn_calibrated_support']),
                   'union_roi_fraction':{m:box_union_area(rows)/(trace['width']*trace['height']) for m,rows in selected.items()}}
            audits.append(audit); save(output/'traces'/(path.stem+'.json'),audit)
            np.savez_compressed(output/(path.stem+'_maps.npz'),features=query,fusion=score)
            save(output/'progress.json',{'status':'running','completed':index,'planned':30,'cases':cases,'threshold':frozen['threshold']})
            print(f'test01 {index}/30 {path.name}: old={len(baseline)} new={len(selected["cnn_calibrated_support"])} added={audit["added_selected_count"]} seconds={audit["total_seconds"]:.1f}',flush=True)
    finally: tiled._merge_candidates,impl.review_components=original_merge,original_review
    def metrics(rows):
        faults=[r for r in rows if r['targets']]; normals=[r for r in rows if not r['targets']]
        single=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'macro_overlap':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals),
                'normal_images_with_candidates':sum(bool(r['candidates']) for r in normals)}
    results={m:{'overall':metrics(rows),'by_kind':{k:metrics([r for r in rows if r['image'].startswith(k+'_')])
                for k in ('damaged','disconnected','misrouted','normal')},'cases':rows} for m,rows in cases.items()}
    save(output/'report.json',{'status':'complete','protocol':'frozen80_cnn_qualified_support_test01_once_v1',
         'threshold':frozen['threshold'],'fit_names':source['fit_names'],'calibration_names':source['calibration_names'],
         'results':results,'audits':audits,'elapsed_seconds':time.perf_counter()-started,'formal_path_changed':False,
         'script_sha256':digest(Path(__file__)),'frozen_rule_sha256':frozen['experiment_sha256'],
         'frozen_source_report_sha256':digest(evidence/'original/report.json'),
         'warning':'Historical test01 baseline already seen, same chassis, not a blind new-scene benchmark. No parameter selection on test01. Normal forced-localization is not full-system specificity.'})
    print(json.dumps({m:{'overall':d['overall'],'by_kind':d['by_kind']} for m,d in results.items()},indent=2))


if __name__=='__main__': main()
