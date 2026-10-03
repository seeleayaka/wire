"""One post-hoc follow-up: swap fused CNN/DINO maps in identical old-pool ranking."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def box_union_area(boxes):
    coords=[tuple(float(b[k]) for k in ('left','top','right','bottom')) for b in boxes]
    if any(not all(np.isfinite(b)) or b[0]>=b[2] or b[1]>=b[3] for b in coords):
        raise ValueError('invalid box geometry')
    edges=sorted({x for b in coords for x in (b[0],b[2])})
    total=0.0
    for left,right in zip(edges,edges[1:]):
        intervals=sorted((b[1],b[3]) for b in coords if b[0]<right and b[2]>left)
        covered=0.0; end=-float('inf')
        for top,bottom in intervals:
            covered+=max(0,bottom-max(top,end)); end=max(end,bottom)
        total+=(right-left)*covered
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--cnn-maps',type=Path,required=True)
    p.add_argument('--dino-maps',type=Path,required=True)
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('use a new ranking directory')
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup,save
    from tools.evaluate_anchored_local import anchored_selection
    from tools.probe_local_normal_bank import region_score
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(a.repo); tiled.LARGE_ROI_MAX_CANDIDATES=6
    cases={m:[] for m in ('baseline','dino_anchor','cnn_anchor')}
    geometry=[]
    records=[]
    for directory in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(directory.glob('*.json')) if f.name!='comparison.json']
    source_hash=hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert len(records)==30 and all(r['split']=='val01' and r['source_sha256']==source_hash for r in records)
    for r in records:
        t=r['trace']; pool=tiled._merge_candidates(copy.deepcopy(t['raw']))
        tiled._annotate_roi_edges(pool,t['width'],t['height'])
        published,_=tiled._publish_candidates(pool,[])
        eligible,_=tiled._display_candidates_for_large_roi(published)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==t['local_candidates']
        selected={'baseline':baseline}
        for mode,directory in (('dino_anchor',a.dino_maps),('cnn_anchor',a.cnn_maps)):
            with np.load(directory/(Path(r['image']).stem+'_maps.npz'),allow_pickle=False) as data: score=data['fusion']
            assert score.shape==(21,28) and np.isfinite(score).all()
            selected[mode]=anchored_selection(baseline,pool,score,t['width'],t['height'],region_score)
            assert len(selected[mode])==len(baseline) and (not baseline or selected[mode][0]==baseline[0])
        x,y,_,_=r['bounds']
        key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
        eligible_keys={key(b) for b in eligible}
        geometry.append({'image':r['image'],**{m:{'union_roi_fraction':box_union_area(rows)/(t['width']*t['height']),
                        'sum_roi_fraction':sum((b['right']-b['left'])*(b['bottom']-b['top']) for b in rows)/(t['width']*t['height']),
                        'outside_original_eligible_count':sum(key(b) not in eligible_keys for b in rows)} for m,rows in selected.items()}})
        for mode,rows in selected.items():
            boxes=[{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in rows]
            cases[mode].append({'image':r['image'],'targets':r['targets'],'candidates':boxes})
    def metrics(rows):
        faults=[r for r in rows if r['targets']]; normals=[r for r in rows if not r['targets']]
        single=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'macro_overlap':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals)}
    results={m:{'overall':metrics(rows),'by_kind':{k:metrics([r for r in rows if r['image'].startswith(k+'_')])
               for k in ('damaged','disconnected','misrouted','normal')},'cases':rows} for m,rows in cases.items()}
    # Matched DINO control must reproduce the previous identity-fusion anchor.
    prior=json.loads((a.dino_maps/'report.json').read_text(encoding='utf-8'))['results']['identity_fusion_anchor']['cases']
    assert cases['dino_anchor']==prior
    a.output.mkdir(parents=True)
    save(a.output/'report.json',{'protocol':'posthoc_same_pool_anchor_p95_feature_swap_v1','posthoc':True,
         'formal_path_changed':False,'baseline_exact':True,'dino_control_exact':True,'results':results,
         'geometry':geometry,
         'source_sha256':source_hash,'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'cnn_report_sha256':hashlib.sha256((a.cnn_maps/'report.json').read_bytes()).hexdigest(),
         'dino_report_sha256':hashlib.sha256((a.dino_maps/'report.json').read_bytes()).hexdigest(),
         'warning':'One frozen post-hoc follow-up after CNN heat failure. Uses old merged pool before eligibility, like prior DINO anchor. No new geometry or budget, but different subset. Repeated same-chassis validation is not field or generalization accuracy.'})
    print(json.dumps({m:{'overall':d['overall'],'by_kind':d['by_kind']} for m,d in results.items()},indent=2))


if __name__=='__main__': main()
