"""One fixed coarse/fine fusion policy: keep strongest existing cross-scale evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def anchored_selection(baseline,pool,score,width,height,region_score):
    if not baseline:
        return []
    anchor=baseline[0]
    def key(b):
        return tuple(b[k] for k in ('left','top','right','bottom'))
    used={key(anchor)}
    result=[anchor]
    for b in sorted(pool,key=lambda b:region_score(b,score,width,height),reverse=True):
        if key(b) not in used:
            result.append(b); used.add(key(b))
        if len(result)>=len(baseline):
            break
    return result[:len(baseline)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--exclude-neighbor-radius',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup,save
    from tools.probe_local_normal_bank import region_score
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(a.repo)
    tiled.LARGE_ROI_MAX_CANDIDATES=6
    records=[]
    for directory in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(directory.glob('*.json')) if f.name!='comparison.json']
    assert len(records)==30 and all(r['split']=='val01' for r in records)
    sha=hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert all(r['source_sha256']==sha for r in records)
    suffix=f'_exclude{a.exclude_neighbor_radius}' if a.exclude_neighbor_radius else ''
    source_report=json.loads((a.maps/(f'report_exclude{a.exclude_neighbor_radius}.json' if suffix else 'report.json')).read_text(encoding='utf-8'))
    cases=[]
    changes=[]
    for r in records:
        trace=r['trace']
        assert not trace['metadata']['repetitive_group_candidate_count']
        pool=tiled._merge_candidates(copy.deepcopy(trace['raw']))
        tiled._annotate_roi_edges(pool,trace['width'],trace['height'])
        published,_=tiled._publish_candidates(pool,[])
        eligible,_=tiled._display_candidates_for_large_roi(published)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==trace['local_candidates']
        with np.load(a.maps/(Path(r['image']).stem+suffix+'_maps.npz')) as data:
            score=data['normal_bank']
        assert tuple(score.shape)==tuple(source_report['grid'][:2])
        selected=anchored_selection(baseline,pool,score,trace['width'],trace['height'],region_score)
        assert len(selected)==len(baseline)
        assert not baseline or selected[0]==baseline[0]
        x,y,_,_=r['bounds']
        boxes=[{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in selected]
        case={'image':r['image'],'targets':r['targets'],'candidates':boxes}
        cases.append(case)
        if r['image'].startswith('normal_'):
            key=lambda rows:{tuple(b[k] for k in ('left','top','right','bottom')) for b in rows}
            changes.append(key(selected)!=key(baseline))
    def metrics(rows):
        faults=[c for c in rows if not c['image'].startswith('normal_')]
        normals=[c for c in rows if c['image'].startswith('normal_')]
        summaries=[summarize([c],'candidates') for c in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
            'image_macro_overlap_fraction':float(np.mean([m['target_boxes_with_overlap']/m['target_boxes'] for m in summaries])) if summaries else None,
            'image_macro_best_iou':float(np.mean([m['mean_best_target_iou'] for m in summaries])) if summaries else None,
            'normal_candidate_count':sum(len(c['candidates']) for c in normals)}
    groups=source_report['groups']
    result={'policy':'One original cross-scale anchor plus novelty-ranked remaining slots; no label-dependent branches.',
        'source_maps':str(a.maps),'exclude_neighbor_radius':a.exclude_neighbor_radius,'overall':metrics(cases),
        'by_kind':{kind:metrics([c for c in cases if c['image'].startswith(kind+'_')]) for kind in ('damaged','disconnected','misrouted','normal')},
        'ordinal_groups':{str(i):metrics([c for c in cases if c['image'] in names]) for i,names in groups.items()},
        'normal_images_with_changed_geometry':sum(changes),'cases':cases,
        'by_image':[{'image':c['image'],**metrics([c])} for c in cases]}
    save(a.output,result)
    print(json.dumps({k:result[k] for k in ('overall','by_kind','normal_images_with_changed_geometry')},indent=2))

if __name__=='__main__':
    main()
