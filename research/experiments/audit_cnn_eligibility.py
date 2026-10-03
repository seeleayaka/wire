"""Fixed gate audit and CNN ranking restricted to production eligibility."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def support_facts(candidate):
    e=candidate['evidence_summary']
    refinement=int(e.get('refinement_tile_count',0))
    primary=int(e.get('primary_tile_count',e.get('source_tile_count',0)))
    whole=bool(e['whole_roi_overlap'])
    cross=float(candidate.get('evidence_scores',{}).get('cross_evidence_pixel_ratio_max',0))>=.50
    cross_scale=refinement>=1 and (whole or primary>=1)
    repeated=refinement>=2
    thin=int(e.get('thin_core_observation_count',0))
    thin_only=thin>0 and thin==int(e.get('merged_observation_count',0))
    allowed=(cross and cross_scale) if thin_only else (whole or primary>=4 or cross and (cross_scale or repeated))
    reason='allowed' if allowed else ('thin_missing_cross_evidence' if thin_only and not cross else
           'thin_missing_cross_scale' if thin_only else 'weak_cross_evidence' if (cross_scale or repeated) and not cross else
           'insufficient_spatial_support')
    return {'allowed':bool(allowed),'reason':reason,'whole':whole,'primary':primary,'refinement':refinement,
            'strong_cross':cross,'thin_only':thin_only,'tile_edge_only':bool(e['tile_edge_only'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('use a new output directory')
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup,save
    from tools.evaluate_anchored_local import anchored_selection
    from tools.probe_local_normal_bank import region_score
    from tools.probe_cnn_candidate_rank import box_union_area
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(a.repo); tiled.LARGE_ROI_MAX_CANDIDATES=6
    records=[]
    for directory in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(directory.glob('*.json')) if f.name!='comparison.json']
    source_hash=hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert len(records)==30 and all(r['split']=='val01' and r['source_sha256']==source_hash for r in records)
    prior=json.loads(a.prior.read_text(encoding='utf-8'))
    expected={m:{c['image']:c for c in prior['results'][m]['cases']} for m in ('baseline','cnn_anchor')}
    cases={m:[] for m in ('baseline','cnn_unrestricted','cnn_original_eligible')}; audits=[]
    key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    for r in records:
        t=r['trace']; pool=tiled._merge_candidates(copy.deepcopy(t['raw']))
        tiled._annotate_roi_edges(pool,t['width'],t['height'])
        published,suppressed=tiled._publish_candidates(pool,[])
        assert not suppressed and not t['metadata']['repetitive_group_candidate_count']
        eligible,rejected=tiled._display_candidates_for_large_roi(published)
        eligible_keys={key(b) for b in eligible}
        assert all(support_facts(b)['allowed']==(key(b) in eligible_keys) for b in published)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==t['local_candidates']
        with np.load(a.maps/(Path(r['image']).stem+'_maps.npz'),allow_pickle=False) as data: score=data['fusion']
        selected={'baseline':baseline,'cnn_unrestricted':anchored_selection(baseline,pool,score,t['width'],t['height'],region_score),
                  'cnn_original_eligible':anchored_selection(baseline,eligible,score,t['width'],t['height'],region_score)}
        x,y,_,_=r['bounds']
        pool_audit=[]
        for b in pool:
            facts=support_facts(b)
            pool_audit.append({'box':key(b),'support':facts,'cnn_region_p95':region_score(b,score,t['width'],t['height']),
                               'selected_unrestricted':key(b) in {key(v) for v in selected['cnn_unrestricted']}})
        audits.append({'image':r['image'],'pool_count':len(pool),'eligible_count':len(eligible),'pool':pool_audit,
                       'union_roi_fraction':{m:box_union_area(rows)/(t['width']*t['height']) for m,rows in selected.items()}})
        for m,rows in selected.items():
            assert len(rows)==len(baseline) and (not rows or rows[0]==baseline[0])
            if m=='cnn_original_eligible': assert all(key(b) in eligible_keys for b in rows)
            boxes=[{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in rows]
            case={'image':r['image'],'targets':r['targets'],'candidates':boxes}
            if m in ('baseline','cnn_unrestricted'): assert case==expected['baseline' if m=='baseline' else 'cnn_anchor'][r['image']]
            cases[m].append(case)
    def metrics(rows):
        faults=[r for r in rows if r['targets']]; normals=[r for r in rows if not r['targets']]
        single=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'macro_overlap':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals)}
    results={m:{'overall':metrics(rows),'by_kind':{k:metrics([r for r in rows if r['image'].startswith(k+'_')])
                for k in ('damaged','disconnected','misrouted','normal')},'cases':rows} for m,rows in cases.items()}
    reason_counts={}
    for group in ('fault','normal'):
        filtered=[b for r in audits if r['image'].startswith('normal_')==(group=='normal') for b in r['pool']
                  if b['selected_unrestricted'] and not b['support']['allowed']]
        reason_counts[group]={reason:sum(b['support']['reason']==reason for b in filtered)
                              for reason in sorted({b['support']['reason'] for b in filtered})}
    a.output.mkdir(parents=True)
    save(a.output/'report.json',{'protocol':'same_cnn_anchor_original_eligibility_v1','formal_path_changed':False,
         'source_sha256':source_hash,'prior_report_sha256':hashlib.sha256(a.prior.read_bytes()).hexdigest(),
         'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'baseline_and_unrestricted_exact':True,'gate_predicates_exact':True,'results':results,
         'rejected_selection_reasons':reason_counts,'audits':audits,
         'warning':'Repeated val01, fragmented source annotations and same chassis; not field accuracy. Eligibility preserved control only, no gate removed or test01 tuning.'})
    print(json.dumps({'rejected_selection_reasons':reason_counts,'results':{m:{'overall':d['overall'],'by_kind':d['by_kind']} for m,d in results.items()}},indent=2))


if __name__=='__main__': main()
