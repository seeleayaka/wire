"""Normal-calibrated seeded heat candidates; frozen exploratory protocol."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np


def peak_boxes(score,threshold,width,height):
    score=np.asarray(score,dtype=np.float64)
    if score.ndim!=2 or not np.isfinite(score).all() or threshold<0 or min(width,height)<=0:
        raise ValueError('invalid heat geometry')
    rows,cols=score.shape
    maxima=(score==cv2.dilate(score,np.ones((3,3),np.uint8))) & (score>threshold)
    count,labels=cv2.connectedComponents(maxima.astype(np.uint8),connectivity=8)
    seeds=[]
    for index in range(1,count):
        positions=np.flatnonzero(labels.ravel()==index)
        chosen=int(positions[np.argmax(score.ravel()[positions])])
        sy,sx=divmod(chosen,cols)
        # A broad flat bridge can contain 3x3 maxima in its middle while
        # touching a higher peak at its ends. Reject that false plateau seed.
        plateau_level=score[sy,sx]; stack=[(sy,sx)]; seen={(sy,sx)}; has_higher=False
        while stack:
            y,x=stack.pop()
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<rows and 0<=nx<cols:
                        if score[ny,nx]>plateau_level:
                            has_higher=True
                        elif score[ny,nx]==plateau_level and (ny,nx) not in seen:
                            seen.add((ny,nx)); stack.append((ny,nx))
        if not has_higher:
            seeds.append((sy,sx))
    seeds.sort(key=lambda point:(-score[point],point))
    occupied=np.zeros(score.shape,dtype=bool)
    result=[]
    for sy,sx in seeds:
        if occupied[sy,sx]:
            continue
        level=max(threshold,0.5*score[sy,sx])
        stack=[(sy,sx)]; cells=[]; seen={(sy,sx)}
        while stack:
            y,x=stack.pop()
            if occupied[y,x] or score[y,x]<level:
                continue
            cells.append((y,x))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<rows and 0<=nx<cols and (ny,nx) not in seen:
                        seen.add((ny,nx)); stack.append((ny,nx))
        if not cells:
            continue
        ys,xs=zip(*cells); occupied[ys,xs]=True
        result.append({'left':int(np.floor(min(xs)*width/cols)),
                       'top':int(np.floor(min(ys)*height/rows)),
                       'right':min(width,int(np.ceil((max(xs)+1)*width/cols))),
                       'bottom':min(height,int(np.ceil((max(ys)+1)*height/rows))),
                       'peak':float(score[sy,sx]),'support_cells':len(cells),
                       'diagnostic_source':'normal_calibrated_peak_growth'})
    return result


def choose(baseline,generated,anchor):
    budget=len(baseline)
    if not budget:
        return []
    output=[baseline[0]] if anchor else []
    used={tuple(b[k] for k in ('left','top','right','bottom')) for b in output}
    for box in generated:
        key=tuple(box[k] for k in ('left','top','right','bottom'))
        if len(output)>=budget:
            break
        if key not in used:
            output.append(box); used.add(key)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    parser.add_argument('--original-maps',type=Path,required=True)
    parser.add_argument('--metric-maps',type=Path,required=True)
    parser.add_argument('--feature-cache',type=Path,required=True)
    parser.add_argument('--fault-cache',type=Path,required=True)
    parser.add_argument('--normal-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new experiment directory')
    sys.path.insert(0,str(args.repo))
    from tools.probe_spatial_metric import unweighted_score
    from tools.probe_spatial_normal import fuse
    from tools.merge_audit import setup,save
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(args.repo); tiled.LARGE_ROI_MAX_CANDIDATES=6
    source=json.loads((args.original_maps/'report.json').read_text(encoding='utf-8'))
    metric=json.loads((args.metric_maps/'report.json').read_text(encoding='utf-8'))
    assert metric['source_report_sha256']==hashlib.sha256((args.original_maps/'report.json').read_bytes()).hexdigest()
    manifest={(e['split'],e['image']):e for e in source['feature_manifest']}
    with np.load(args.original_maps/'spatial_model.npz',allow_pickle=False) as data:
        model={key:data[key] for key in ('channels','mean','precision')}
    with np.load(args.original_maps/'calibration_maps.npz',allow_pickle=False) as data:
        local_cal=data['local']
    peaks=[]
    for local,name in zip(local_cal,source['calibration_names']):
        entry=manifest['train01',name]; path=args.feature_cache/entry['cache']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['cache_sha256']
        with np.load(path,allow_pickle=False) as data:
            identity=unweighted_score(data['features'],model)
        fused=fuse(local,identity,source['calibration']['local_scale'],metric['calibration']['identity_scale'])
        peaks.append(float(fused.max()))
    threshold=float(np.percentile(peaks,95))
    # Protocol fixed before reading validation outcomes:
    # 3x3 local maxima; 8-connected grow >= max(normal threshold, half peak).
    records=[]
    for directory in (args.fault_cache,args.normal_cache):
        records += [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('*.json'))
                    if p.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' for r in records)
    source_hash=hashlib.sha256((args.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert source_hash==source['source_sha256'] and all(r['source_sha256']==source_hash for r in records)
    cases={mode:[] for mode in ('baseline','generated_only','anchor_generated')}
    audits=[]
    for record in records:
        trace=record['trace']
        pool=tiled._merge_candidates(copy.deepcopy(trace['raw']))
        tiled._annotate_roi_edges(pool,trace['width'],trace['height'])
        published,_=tiled._publish_candidates(pool,[])
        eligible,_=tiled._display_candidates_for_large_roi(published)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==trace['local_candidates']
        with np.load(args.metric_maps/(Path(record['image']).stem+'_maps.npz'),allow_pickle=False) as data:
            generated=peak_boxes(data['fusion'],threshold,trace['width'],trace['height'])
        # No source label is consulted above this point.
        left,top,_,_=record['bounds']
        for mode in cases:
            selected=baseline if mode=='baseline' else choose(baseline,generated,mode=='anchor_generated')
            assert len(selected)<=len(baseline)
            boxes=[{**b,'left':b['left']+left,'right':b['right']+left,
                    'top':b['top']+top,'bottom':b['bottom']+top} for b in selected]
            cases[mode].append({'image':record['image'],'targets':record['targets'],'candidates':boxes})
        all_boxes=[{**b,'left':b['left']+left,'right':b['right']+left,
                   'top':b['top']+top,'bottom':b['bottom']+top} for b in generated]
        audits.append({'image':record['image'],'precap_count':len(generated),
                       'precap_metrics':summarize([{'targets':record['targets'],'candidates':all_boxes}],'candidates') if record['targets'] else None,
                       'generated_local_boxes':generated})

    def metrics(rows):
        faults=[r for r in rows if not r['image'].startswith('normal_')]
        normals=[r for r in rows if r['image'].startswith('normal_')]
        per_image=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'macro_overlap':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in per_image])) if faults else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals),
                'normal_images_with_candidates':sum(bool(r['candidates']) for r in normals)}
    result={mode:{'overall':metrics(rows),'by_kind':{kind:metrics([r for r in rows if r['image'].startswith(kind+'_')])
            for kind in ('damaged','disconnected','misrouted','normal')},'cases':rows} for mode,rows in cases.items()}
    args.output.mkdir(parents=True)
    save(args.output/'report.json',{'protocol':'normal_max95_seeded_half_peak_v1','threshold':threshold,
         'normal_calibration_maxima':peaks,'calibration_names':source['calibration_names'],
         'growth_peak_ratio':0.5,'maxima_window':3,'connectivity':8,'baseline_exact':True,
         'source_sha256':source_hash,'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'metric_report_sha256':hashlib.sha256((args.metric_maps/'report.json').read_bytes()).hexdigest(),
         'results':result,'generated_audits':audits,'formal_path_changed':False,
         'warning':'Exploratory coarse-grid boxes, repeated validation development, incomplete fragmented annotations. Normal forced-localization counts do not measure full-cascade or field false positives.'})
    print(json.dumps({'threshold':threshold,'results':{m:{'overall':d['overall'],'by_kind':d['by_kind']} for m,d in result.items()}},indent=2))


if __name__=='__main__':
    main()
