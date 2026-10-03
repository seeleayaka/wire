"""Fixed local normal-bank probe: training normals only, validation diagnostics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

def local_distance(query, bank, k=3, radius=1):
    """Mean distance to k different normal images; each may match nearby patches."""
    query=np.asarray(query,dtype=np.float32)
    bank=np.asarray(bank,dtype=np.float32)
    if query.ndim!=3 or bank.ndim!=4 or bank.shape[1:]!=query.shape:
        raise ValueError('feature geometry mismatch')
    if min(query.shape)<=0:
        raise ValueError('empty feature geometry')
    if not 1<=k<=len(bank) or radius<0:
        raise ValueError('invalid k or radius')
    if not np.isfinite(query).all() or not np.isfinite(bank).all():
        raise ValueError('nonfinite feature')
    query=query/np.maximum(np.linalg.norm(query,axis=-1,keepdims=True),1e-8)
    bank=bank/np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-8)
    n,h,w,_=bank.shape
    best=np.full((n,h,w),-1,dtype=np.float32)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            ya,yb=max(0,-dy),min(h,h-dy)
            xa,xb=max(0,-dx),min(w,w-dx)
            if ya>=yb or xa>=xb:
                continue
            similarities=np.einsum('nhwc,hwc->nhw',bank[:,ya+dy:yb+dy,xa+dx:xb+dx],query[ya:yb,xa:xb],optimize=True)
            best[:,ya:yb,xa:xb]=np.maximum(best[:,ya:yb,xa:xb],similarities)
    distances=np.clip(1-best,0,2)
    return np.partition(distances,k-1,axis=0)[:k].mean(axis=0)

def region_score(box,score,width,height):
    h,w=score.shape
    x0=max(0,min(w-1,int(np.floor(box['left']*w/width))))
    y0=max(0,min(h-1,int(np.floor(box['top']*h/height))))
    x1=max(x0+1,min(w,int(np.ceil(box['right']*w/width))))
    y1=max(y0+1,min(h,int(np.ceil(box['bottom']*h/height))))
    return float(np.percentile(score[y0:y1,x0:x1],95))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude-neighbor-radius',type=int,default=0,
                   help='Predefined filename-number proximity stress test, not true scene groups')
    p.add_argument('--feature-max-edge',type=int,default=392)
    p.add_argument('--patch-radius',type=int,default=1)
    a=p.parse_args()
    if a.exclude_neighbor_radius<0 or a.feature_max_edge<224 or a.patch_radius<0:
        raise ValueError('invalid geometry or exclusion radius')
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup,save
    from tools.evaluate_mendeley_local_refinement import summarize
    impl,tiled=setup(a.repo)
    from evaluate_mendeley_balanced import read_image
    import dino_feature_diff as feature_module
    feature_module.MAX_EDGE=a.feature_max_edge
    extract_features=feature_module.extract_features
    import assembly_auto_review_robust_v3 as perspective
    dataset=a.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    records=[]
    for directory in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(directory.glob('*.json')) if f.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' for r in records)
    bounds=records[0]['bounds']
    assert all(r['bounds']==bounds for r in records)
    sha=hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert all(r['source_sha256']==sha for r in records)
    a.output.mkdir(parents=True,exist_ok=True)
    reference=read_image(dataset/'images/train01/normal_073.JPG')
    left,top,right,bottom=bounds
    # The existing feature cache key omits resolution, so isolate this reference.
    reference_features,_=extract_features(reference[top:bottom,left:right],cache_reference=False)
    reference_identity=hashlib.sha256(reference.tobytes()).hexdigest()
    def features(path):
        # File identity includes canonical reference and crop geometry.
        identity=hashlib.sha256((str(path.resolve())+str(path.stat().st_size)+str(path.stat().st_mtime_ns)+str(bounds)+str(reference_features.shape)+reference_identity+'local_bank_v1').encode()).hexdigest()
        cached=a.output/(identity+'.npz')
        if cached.is_file():
            with np.load(cached) as data:
                return data['features'],True
        aligned,_=perspective.automatic_homography(reference,read_image(path))
        if aligned is None:
            raise RuntimeError(f'cannot align {path.name}')
        value,_=extract_features(aligned[top:bottom,left:right],cache_reference=False)
        np.savez_compressed(cached,features=value)
        return value,False
    bank=[]
    bank_names=[]
    normal_paths=sorted((dataset/'images/train01').glob('normal_*.JPG'))
    assert len(normal_paths)==120
    for index,path in enumerate(normal_paths,1):
        value,hit=features(path)
        assert value.shape==reference_features.shape
        bank.append(value); bank_names.append(path.name)
        if index%10==0:
            print(f'train bank {index}/120 cache={hit}',flush=True)
    bank=np.stack(bank)
    modes=('baseline','single_reference_rank','normal_bank_rank','normal_bank_all_merged')
    cases={mode:[] for mode in modes}
    maps=[]
    tiled.LARGE_ROI_MAX_CANDIDATES=6
    start=time.monotonic()
    for index,r in enumerate(records,1):
        q,hit=features(dataset/'images/val01'/r['image'])
        # Check source membership to prohibit query leakage into the normal bank.
        assert r['image'] not in bank_names
        single=local_distance(q,reference_features[None],k=1,radius=a.patch_radius)
        if a.exclude_neighbor_radius:
            number=int(Path(r['image']).stem.rsplit('_',1)[1])
            keep=[abs(int(Path(name).stem.rsplit('_',1)[1])-number)>a.exclude_neighbor_radius for name in bank_names]
            query_bank=bank[np.asarray(keep)]
        else:
            query_bank=bank
        novelty=local_distance(q,query_bank,k=3,radius=a.patch_radius)
        trace=r['trace']
        assert not trace['metadata']['repetitive_group_candidate_count']
        merged=tiled._merge_candidates(copy.deepcopy(trace['raw']))
        tiled._annotate_roi_edges(merged,trace['width'],trace['height'])
        published,_=tiled._publish_candidates(merged,[])
        eligible,_=tiled._display_candidates_for_large_roi(published)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==trace['local_candidates']
        for mode in modes:
            if mode=='baseline':
                selected=baseline
            else:
                pool=merged if mode=='normal_bank_all_merged' else eligible
                score=single if mode=='single_reference_rank' else novelty
                selected=sorted(pool,key=lambda b:region_score(b,score,trace['width'],trace['height']),reverse=True)[:len(baseline)]
            global_boxes=[{**b,'left':b['left']+left,'right':b['right']+left,'top':b['top']+top,'bottom':b['bottom']+top} for b in selected]
            cases[mode].append({'image':r['image'],'targets':r['targets'],'candidates':global_boxes})
        maps.append({'image':r['image'],'reference_count':len(query_bank),'single_p95':float(np.percentile(single,95)),
                     'bank_p95':float(np.percentile(novelty,95)), 'bank_max':float(novelty.max())})
        suffix=f'_exclude{a.exclude_neighbor_radius}' if a.exclude_neighbor_radius else ''
        np.savez_compressed(a.output/(Path(r['image']).stem+suffix+'_maps.npz'),single=single,normal_bank=novelty)
        print(f'val {index}/30 {r["image"]} cache={hit}',flush=True)
    def metrics(rows):
        faults=[c for c in rows if not c['image'].startswith('normal_')]
        normals=[c for c in rows if c['image'].startswith('normal_')]
        return {'faults':summarize(faults,'candidates') if faults else None,'normal_candidate_count':sum(len(c['candidates']) for c in normals)}
    groups={i:[] for i in range(3)}
    for kind in ('damaged','disconnected','misrouted','normal'):
        names=sorted(r['image'] for r in records if r['image'].startswith(kind+'_'))
        for idx,name in enumerate(names):
            groups[min(2,idx*3//len(names))].append(name)
    results={mode:{'overall':metrics(rows),'by_kind':{kind:metrics([c for c in rows if c['image'].startswith(kind+'_')]) for kind in ('damaged','disconnected','misrouted','normal')},
        'ordinal_groups':{str(i):metrics([c for c in rows if c['image'] in names]) for i,names in groups.items()},'cases':rows} for mode,rows in cases.items()}
    report_name=f'report_exclude{a.exclude_neighbor_radius}.json' if a.exclude_neighbor_radius else 'report.json'
    save(a.output/report_name,{'split':'val01','bank_split':'train01','bank_names':bank_names,'k':3,'radius_patches':a.patch_radius,
        'feature_max_edge':a.feature_max_edge,
        'exclude_neighbor_radius':a.exclude_neighbor_radius,
        'grid':list(reference_features.shape),'baseline_exact':True,'groups':groups,'maps':maps,'results':results,
        'warning':'Coarse patch features and correlated scenes; fixed-budget normal counts are equal by construction. No calibrated local fault threshold or unseen-scene claim.',
        'elapsed_validation_seconds':time.monotonic()-start})
    print(json.dumps({mode:r['overall'] for mode,r in results.items()},indent=2))

if __name__=='__main__':
    main()
