"""One-factor diagnostic: remove covariance weighting, keep all other choices."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def unweighted_score(query, model):
    # Same full-vector normalization, selected channels and spatial mean as
    # the original model. Only the precision matrix is replaced by identity.
    value=np.asarray(query,dtype=np.float64)
    if value.ndim!=3 or value.shape[:2]!=model['mean'].shape[:2] or not np.isfinite(value).all():
        raise ValueError('invalid feature geometry')
    value=value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-8)
    delta=value[...,model['channels']]-model['mean']
    return np.sqrt(np.mean(delta**2,axis=-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--feature-cache',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--normal-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new diagnostic output')
    sys.path.insert(0,str(args.repo))
    from tools.probe_spatial_normal import spatial_score, normal_scale, fuse
    from tools.audit_spatial_evidence import target_grid,rank_auc
    from tools.evaluate_anchored_local import anchored_selection
    from tools.probe_local_normal_bank import region_score
    from tools.merge_audit import setup,save
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(args.repo); tiled.LARGE_ROI_MAX_CANDIDATES=6
    source=json.loads((args.maps/'report.json').read_text(encoding='utf-8'))
    with np.load(args.maps/'spatial_model.npz',allow_pickle=False) as data:
        model={key:data[key] for key in ('channels','mean','precision')}
    manifest={(m['split'],m['image']):m for m in source['feature_manifest']}

    def features(split,name):
        entry=manifest[split,name]
        path=args.feature_cache/entry['cache']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['cache_sha256']:
            raise ValueError('cached feature changed')
        with np.load(path,allow_pickle=False) as data:
            return data['features']

    calibration=[unweighted_score(features('train01',name),model) for name in source['calibration_names']]
    scale,calibration_info=normal_scale(calibration)
    local_scale=source['calibration']['local_scale']
    with np.load(args.maps/'calibration_maps.npz',allow_pickle=False) as data:
        local_cal=data['local']
    fused_cal=[fuse(l,p,local_scale,scale) for l,p in zip(local_cal,calibration)]
    fusion_scale,fusion_info=normal_scale(fused_cal)
    records=[]
    for directory in (args.fault_cache,args.normal_cache):
        records += [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('*.json'))
                    if p.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' for r in records)
    source_hash=hashlib.sha256((args.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert source_hash==source['source_sha256'] and all(r['source_sha256']==source_hash for r in records)
    args.output.mkdir(parents=True)
    cases={mode:[] for mode in ('identity_anchor','identity_fusion_anchor')}
    rows=[]
    for r in records:
        name=r['image']; query=features('val01',name)
        with np.load(args.maps/(Path(name).stem+'_maps.npz'),allow_pickle=False) as data:
            old_position,local=data['position'],data['local']
        np.testing.assert_allclose(spatial_score(query,model),old_position,atol=1e-7,rtol=1e-7)
        position=unweighted_score(query,model)
        fusion=fuse(local,position,local_scale,scale)
        trace=r['trace']; bounds=r['bounds']
        pool=tiled._merge_candidates(copy.deepcopy(trace['raw']))
        tiled._annotate_roi_edges(pool,trace['width'],trace['height'])
        published,_=tiled._publish_candidates(pool,[])
        eligible,_=tiled._display_candidates_for_large_roi(published)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==trace['local_candidates']
        for mode,score in (('identity_anchor',position),('identity_fusion_anchor',fusion)):
            selected=anchored_selection(baseline,pool,score,trace['width'],trace['height'],region_score)
            assert len(selected)==len(baseline) and (not baseline or selected[0]==baseline[0])
            left,top,_,_=bounds
            boxes=[{**b,'left':b['left']+left,'right':b['right']+left,
                    'top':b['top']+top,'bottom':b['bottom']+top} for b in selected]
            cases[mode].append({'image':name,'targets':r['targets'],'candidates':boxes})
        mask=target_grid(r['targets'],bounds,position.shape)
        rows.append({'image':name,'mahal_grid_auc':rank_auc(old_position,mask),
                     'identity_grid_auc':rank_auc(position,mask),'fusion_grid_auc':rank_auc(fusion,mask),
                     'identity_image_normalized':float(np.percentile(position,95)/scale),
                     'fusion_image_normalized':float(np.percentile(fusion,95)/fusion_scale)})
        np.savez_compressed(args.output/(Path(name).stem+'_maps.npz'),identity=position,fusion=fusion)

    def metrics(selected):
        faults=[r for r in selected if not r['image'].startswith('normal_')]
        normals=[r for r in selected if r['image'].startswith('normal_')]
        single=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'image_macro_overlap_fraction':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals)}
    result={mode:{'overall':metrics(selected),'by_kind':{kind:metrics([r for r in selected if r['image'].startswith(kind+'_')])
           for kind in ('damaged','disconnected','misrouted','normal')},'cases':selected} for mode,selected in cases.items()}
    auc={}
    for group in ('all','damaged','disconnected','misrouted'):
        selected=[r for r in rows if not r['image'].startswith('normal_') and (group=='all' or r['image'].startswith(group+'_'))]
        auc[group]={key:float(np.mean([r[key] for r in selected])) for key in ('mahal_grid_auc','identity_grid_auc','fusion_grid_auc')}
    normal_response={key:{'normal_above_scale':sum(r[key]>1 for r in rows if r['image'].startswith('normal_')),
                         'fault_above_scale':sum(r[key]>1 for r in rows if not r['image'].startswith('normal_'))}
                     for key in ('identity_image_normalized','fusion_image_normalized')}
    save(args.output/'report.json',{'diagnostic':'replace precision by identity; no channel, reference, mean, budget or fusion policy change',
         'source_report_sha256':hashlib.sha256((args.maps/'report.json').read_bytes()).hexdigest(),
         'mahal_map_replay_exact':True,'formal_path_changed':False,'results':result,'grid_auc':auc,
         'normal_response':normal_response,'evidence':rows,'calibration':{'identity_scale':scale,'identity':calibration_info,
         'fusion_scale':fusion_scale,'fusion':fusion_info},'historical_baseline':source['results']['baseline']['overall'],
         'warning':'Exploratory diagnosis on repeatedly inspected val01; source-box grid labels are not pixel truth. Image thresholds are not deployed field accuracy.'})
    print(json.dumps({'results':{m:{'overall':r['overall'],'by_kind':r['by_kind']} for m,r in result.items()},
                      'grid_auc':auc,'normal_response':normal_response},indent=2))


if __name__=='__main__':
    main()
