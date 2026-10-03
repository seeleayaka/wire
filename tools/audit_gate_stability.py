"""Fixed-policy validation stress test; ordinal buckets are not unseen scenes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup, publish
    from tools.evaluate_mendeley_local_refinement import summarize, iou
    _, tiled = setup(a.repo)
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    records = []
    for folder in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(folder.glob('*.json')) if f.name!='comparison.json']
    assert len(records)==30 and len({r['image'] for r in records})==30
    sha = hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert all(r['split']=='val01' and r['source_sha256']==sha for r in records)
    modes = ('baseline','dino_rank','relaxed_two_primary','relaxed_two_primary_dino')
    cases = {mode:[] for mode in modes}
    attribution = []
    for r in records:
        trace = r['trace']
        assert not trace['metadata']['repetitive_group_candidate_count']
        merged = tiled._merge_candidates(copy.deepcopy(trace['raw']))
        assert publish(copy.deepcopy(merged),trace,tiled)==trace['local_candidates']
        tiled._annotate_roi_edges(merged,trace['width'],trace['height'])
        published,_ = tiled._publish_candidates(merged,[])
        eligible,suppressed = tiled._display_candidates_for_large_roi(published)
        budget,_ = tiled._large_roi_candidate_budget(eligible)
        dx,dy,_,_ = r['bounds']
        def globalize(rows):
            return [{**b,'left':b['left']+dx,'right':b['right']+dx,'top':b['top']+dy,'bottom':b['bottom']+dy} for b in rows]
        def extra_supported(b):
            e = b['evidence_summary']
            return (e['primary_tile_count']>=2 and not e['tile_edge_only']
                and e.get('thin_core_observation_count',0)!=e['merged_observation_count']
                and b.get('evidence_scores',{}).get('cross_evidence_pixel_ratio_max',0)>=.5)
        extra = [b for b in suppressed if extra_supported(b)]
        for mode in modes:
            pool = eligible+extra if mode.startswith('relaxed') else eligible
            fixed_budget = budget if eligible else min(6,len(pool))
            if 'dino' in mode:
                selected = sorted(pool,key=lambda b:b.get('evidence_scores',{}).get('dino_mean_max',0),reverse=True)[:fixed_budget]
            else:
                selected,_ = tiled._limit_candidates_for_large_roi(pool,budget=fixed_budget)
            if mode=='baseline':
                assert selected==trace['local_candidates']
            old_coords={tuple(b[k] for k in ('left','top','right','bottom')) for b in trace['local_candidates']}
            new_coords={tuple(b[k] for k in ('left','top','right','bottom')) for b in selected}
            cases[mode].append({'image':r['image'],'targets':r['targets'],'candidates':globalize(selected),
                'eligible_count':len(pool),'selected_geometry_changed':old_coords!=new_coords})
        # Only source-box overlap attribution, not verified false-positive labels.
        gs = globalize(suppressed)
        ge = globalize(eligible)
        lost = [t for t in r['targets'] if any(iou(b,t)>0 for b in gs) and not any(iou(b,t)>0 for b in ge)]
        attribution.append({'image':r['image'],'suppressed':len(suppressed),'newly_eligible':len(extra),
            'targets_covered_only_by_suppressed':len(lost),
            'suppressed_whole_roi':sum(b['evidence_summary']['whole_roi_overlap'] for b in suppressed),
            'suppressed_edge_only':sum(b['evidence_summary']['tile_edge_only'] for b in suppressed),
            'suppressed_fewer_than_four_primary':sum(b['evidence_summary']['primary_tile_count']<4 for b in suppressed)})
    def metrics(rows):
        faults=[c for c in rows if not c['image'].startswith('normal_')]
        normals=[c for c in rows if c['image'].startswith('normal_')]
        return {'faults':summarize(faults,'candidates') if faults else None,
            'normal_images':len(normals),'normal_images_with_candidates':sum(bool(c['candidates']) for c in normals),
            'normal_candidate_count':sum(len(c['candidates']) for c in normals),
            'normal_eligible_count':sum(c['eligible_count'] for c in normals),
            'normal_images_with_changed_candidates':sum(c['selected_geometry_changed'] for c in normals)}
    # Consecutive filename-number strata inside each category, fixed before metrics.
    groups = {i:[] for i in range(3)}
    for kind in ('damaged','disconnected','misrouted','normal'):
        names=sorted(r['image'] for r in records if r['image'].startswith(kind+'_'))
        for idx,name in enumerate(names):
            groups[min(2,idx*3//len(names))].append(name)
    results={mode:{'overall':metrics(rows),
        'by_kind':{kind:metrics([c for c in rows if c['image'].startswith(kind+'_')]) for kind in ('damaged','disconnected','misrouted','normal')},
        'ordinal_groups':{str(i):metrics([c for c in rows if c['image'] in names]) for i,names in groups.items()},
        'by_image':[{'image':c['image'],**metrics([c])} for c in rows]} for mode,rows in cases.items()}
    result={'split':'val01','baseline_exact':True,'groups':groups,
        'protocol':'Four fixed mechanism-based policies; same reference and filters. Group statistics are descriptive stability checks, not held-out generalization. Normal localization is forced, bypassing the existing image-level gate.',
        'results':results,'attribution':attribution}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({m:r['overall'] for m,r in results.items()},indent=2))

if __name__=='__main__':
    main()
