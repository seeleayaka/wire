"""Validation-only stage attribution; overlap labels are diagnostic, not fault truth."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('E:/PythonProject10'))
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from tools.merge_audit import setup, publish
    from tools.evaluate_mendeley_local_refinement import iou, summarize
    _, tiled = setup(a.repo)
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    records = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(a.cache.glob('*.json')) if f.name != 'comparison.json']
    assert len(records) == 15 and all(r['split'] == 'val01' for r in records)
    source_hash = hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert all(r['source_sha256'] == source_hash for r in records)
    cases, transitions, feature_rows = [], [], []
    for r in records:
        trace = r['trace']
        raw = copy.deepcopy(trace['raw'])
        merged = tiled._merge_candidates(copy.deepcopy(raw))
        assert publish(copy.deepcopy(merged), trace, tiled) == trace['local_candidates']
        dx, dy, _, _ = r['bounds']
        def globalize(rows):
            return [{**b, 'left':b['left']+dx, 'right':b['right']+dx,
                     'top':b['top']+dy, 'bottom':b['bottom']+dy} for b in rows]
        stages = {'raw':globalize(raw), 'merged':globalize(merged), 'final':r['candidates']}
        eligible = copy.deepcopy(merged)
        tiled._annotate_roi_edges(eligible, trace['width'], trace['height'])
        eligible, _ = tiled._publish_candidates(eligible, [])
        eligible, _ = tiled._display_candidates_for_large_roi(eligible)
        stages['eligible'] = globalize(eligible)
        count = len(trace['local_candidates'])
        for mode in ('dino_rank', 'scale_then_dino'):
            def rank(b):
                score = b.get('evidence_scores', {}).get('dino_mean_max', 0)
                return (b['evidence_summary']['evidence_scale_count'], score) if mode == 'scale_then_dino' else (score,)
            stages[mode] = globalize(sorted(eligible, key=rank, reverse=True)[:count])
        targets = r['targets']
        hits = {k:[any(iou(b,t)>0 for b in boxes) for t in targets] for k,boxes in stages.items()}
        transitions.append({'image':r['image'],
            'uncovered_even_by_all_raw':sum(not h for h in hits['raw']),
            'raw_hit_final_miss':sum(x and not y for x,y in zip(hits['raw'],hits['final'])),
            'final_hit_without_raw':sum(y and not x for x,y in zip(hits['raw'],hits['final'])),
            'merged_hit_final_miss':sum(x and not y for x,y in zip(hits['merged'],hits['final']))})
        cases.append({'image':r['image'], 'targets':targets, **stages})
        for b,g in zip(raw,stages['raw']):
            feature_rows.append({'image':r['image'], 'overlap':any(iou(g,t)>0 for t in targets),
                'difference_score':b['difference_score'], **b.get('evidence_scores',{})})
    # Pairwise AUC: descriptive only, higher value predicts annotation intersection.
    # Within-image aggregation avoids comparing uncalibrated absolute scores across images.
    feature_stats = {}
    for key in ('difference_score','dino_mean','traditional_mean','agreement_mean','cross_evidence_pixel_ratio'):
        wins = pairs = 0
        per_image = []
        for r in records:
            rows = [x for x in feature_rows if x['image']==r['image'] and key in x]
            pos = [x[key] for x in rows if x['overlap']]
            neg = [x[key] for x in rows if not x['overlap']]
            n = len(pos)*len(neg)
            w = sum((x>y)+0.5*(x==y) for x in pos for y in neg)
            wins += w; pairs += n
            per_image.append({'image':r['image'],'auc':w/n if n else None,'positive':len(pos),'negative':len(neg)})
        feature_stats[key] = {'within_image_pair_weighted_auc':wins/pairs if pairs else None,'by_image':per_image}
    result = {'split':'val01','baseline_exact':True,
        'warning':'Annotation intersection is not verified fault identity; all-raw is unlimited-budget diagnostic, not deployable accuracy.',
        'stages':{k:summarize(cases,k) for k in ('raw','merged','eligible','final','dino_rank','scale_then_dino')},
        'transitions':transitions,
        'transition_totals':{k:sum(r[k] for r in transitions) for k in transitions[0] if k!='image'},
        'feature_stats':feature_stats,
        'ranking_by_image':[{'image':c['image'], **{k:summarize([c],k) for k in ('final','dino_rank')}} for c in cases],
        'by_kind':{kind:{k:summarize([c for c in cases if c['image'].startswith(kind)],k) for k in ('raw','merged','final')} for kind in ('damaged','disconnected','misrouted')}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('stages','transition_totals')},indent=2))
    print(json.dumps({k:v['within_image_pair_weighted_auc'] for k,v in feature_stats.items()},indent=2))

if __name__ == '__main__':
    main()
