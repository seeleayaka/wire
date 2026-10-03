"""Normal-calibrated CNN support exception, retaining old spatial requirements."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def qualifies(candidate,cnn_score,threshold):
    from tools.audit_cnn_eligibility import support_facts
    if not np.isfinite(cnn_score) or not np.isfinite(threshold) or threshold<0:
        raise ValueError('invalid calibrated score')
    facts=support_facts(candidate)
    if facts['allowed']: return True
    cross_scale=facts['refinement']>=1 and (facts['whole'] or facts['primary']>=1)
    spatial=cross_scale if facts['thin_only'] else cross_scale or facts['refinement']>=2
    return bool(spatial and cnn_score>threshold)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--eligibility-report',type=Path,required=True)
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('use a new support output')
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup,save
    from tools.evaluate_anchored_local import anchored_selection
    from tools.probe_local_normal_bank import region_score
    from tools.probe_cnn_candidate_rank import box_union_area
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(a.repo); tiled.LARGE_ROI_MAX_CANDIDATES=6
    source=json.loads((a.evidence/'original/report.json').read_text(encoding='utf-8'))
    with np.load(a.evidence/'original/calibration_maps.npz',allow_pickle=False) as data:
        calibration=data['fusion']
    maxima=calibration.max(axis=(1,2))
    threshold=float(np.percentile(maxima,95))
    assert calibration.shape==(30,21,28) and np.isfinite(calibration).all()
    assert not set(source['fit_names']) & set(source['calibration_names'])
    prior=json.loads(a.eligibility_report.read_text(encoding='utf-8'))
    expected={c['image']:c for c in prior['results']['cnn_original_eligible']['cases']}
    records=[]
    for directory in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(directory.glob('*.json')) if f.name!='comparison.json']
    source_hash=hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert len(records)==30 and source_hash==source['source_sha256'] and all(r['split']=='val01' and r['source_sha256']==source_hash for r in records)
    cases={m:[] for m in ('cnn_original_eligible','cnn_calibrated_support')}; audits=[]
    key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    for r in records:
        t=r['trace']; pool=tiled._merge_candidates(copy.deepcopy(t['raw']))
        tiled._annotate_roi_edges(pool,t['width'],t['height'])
        published,suppressed=tiled._publish_candidates(pool,[])
        assert not suppressed
        eligible,_=tiled._display_candidates_for_large_roi(published)
        eligible_keys={key(b) for b in eligible}
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==t['local_candidates']
        with np.load(a.evidence/'metric'/(Path(r['image']).stem+'_maps.npz'),allow_pickle=False) as data: score=data['fusion']
        qualified=[b for b in published if qualifies(b,region_score(b,score,t['width'],t['height']),threshold)]
        selected={'cnn_original_eligible':anchored_selection(baseline,eligible,score,t['width'],t['height'],region_score),
                  'cnn_calibrated_support':anchored_selection(baseline,qualified,score,t['width'],t['height'],region_score)}
        audits.append({'image':r['image'],'qualified_count':len(qualified),'original_eligible_count':len(eligible),
                       'added_qualified_count':sum(key(b) not in eligible_keys for b in qualified),
                       'added_selected_count':sum(key(b) not in eligible_keys for b in selected['cnn_calibrated_support']),
                       'union_roi_fraction':{m:box_union_area(rows)/(t['width']*t['height']) for m,rows in selected.items()}})
        x,y,_,_=r['bounds']
        for m,rows in selected.items():
            assert len(rows)==len(baseline) and (not rows or rows[0]==baseline[0])
            boxes=[{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in rows]
            case={'image':r['image'],'targets':r['targets'],'candidates':boxes}
            if m=='cnn_original_eligible': assert case==expected[r['image']]
            cases[m].append(case)
    def metrics(rows):
        faults=[r for r in rows if r['targets']]; normals=[r for r in rows if not r['targets']]
        single=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'macro_overlap':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals)}
    results={m:{'overall':metrics(rows),'by_kind':{k:metrics([r for r in rows if r['image'].startswith(k+'_')])
                for k in ('damaged','disconnected','misrouted','normal')},'cases':rows} for m,rows in cases.items()}
    a.output.mkdir(parents=True)
    save(a.output/'report.json',{'protocol':'old_spatial_support_cnn_normal_max95_exception_v1','threshold':threshold,
         'normal_calibration_maxima':maxima.tolist(),'calibration_names':source['calibration_names'],
         'original_eligible_control_exact':True,'results':results,'audits':audits,'formal_path_changed':False,
         'source_sha256':source_hash,'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'evidence_report_sha256':hashlib.sha256((a.evidence/'original/report.json').read_bytes()).hexdigest(),
         'calibration_maps_sha256':hashlib.sha256((a.evidence/'original/calibration_maps.npz').read_bytes()).hexdigest(),
         'warning':'Conservative experiment, no theoretical coverage guarantee from interpolated 30-normal percentile. Repeated same-chassis val01 is not field accuracy. CNN does not replace spatial support or identify cable identity.'})
    print(json.dumps({'threshold':threshold,'results':{m:{'overall':d['overall'],'by_kind':d['by_kind']} for m,d in results.items()},
                     'added_by_group':{g:sum(r['added_selected_count'] for r in audits if r['image'].startswith('normal_')==(g=='normal')) for g in ('fault','normal')}},indent=2))


if __name__=='__main__': main()
