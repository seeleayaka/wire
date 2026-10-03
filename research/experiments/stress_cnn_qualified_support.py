"""Two frozen normal-bank halves; sensitivity check, not a new-scene holdout."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--normal-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('use a new stress directory')
    sys.path.insert(0,str(a.repo))
    from tools.merge_audit import setup,save
    from tools.prepare_fine_heat_evidence import mean_model
    from tools.probe_spatial_metric import unweighted_score
    from tools.probe_local_normal_bank import local_distance,region_score
    from tools.probe_spatial_normal import normal_scale,fuse
    from tools.evaluate_anchored_local import anchored_selection
    from tools.probe_cnn_qualified_support import qualifies
    from tools.evaluate_mendeley_local_refinement import summarize
    _,tiled=setup(a.repo); tiled.LARGE_ROI_MAX_CANDIDATES=6
    source=json.loads((a.evidence/'original/report.json').read_text(encoding='utf-8'))
    manifest={(m['split'],m['image']):m for m in source['feature_manifest']}
    features={}
    for key,entry in manifest.items():
        path=a.evidence/'features'/entry['cache']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['cache_sha256']
        with np.load(path,allow_pickle=False) as data: value=data['features']
        assert value.shape==(21,28,192) and np.isfinite(value).all()
        features[key]=value
    records=[]
    for directory in (a.fault_cache,a.normal_cache):
        records += [json.loads(f.read_text(encoding='utf-8')) for f in sorted(directory.glob('*.json')) if f.name!='comparison.json']
    source_hash=hashlib.sha256((a.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert len(records)==30 and source_hash==source['source_sha256'] and all(r['split']=='val01' and r['source_sha256']==source_hash for r in records)
    traces=[]
    for r in records:
        t=r['trace']; pool=tiled._merge_candidates(copy.deepcopy(t['raw']))
        tiled._annotate_roi_edges(pool,t['width'],t['height'])
        eligible,_=tiled._display_candidates_for_large_roi(pool)
        budget,_=tiled._large_roi_candidate_budget(eligible)
        baseline,_=tiled._limit_candidates_for_large_roi(eligible,budget=budget)
        assert baseline==t['local_candidates'] and not t['metadata']['repetitive_group_candidate_count']
        traces.append((r,pool,baseline))
    def metrics(rows):
        faults=[r for r in rows if r['targets']]; normals=[r for r in rows if not r['targets']]
        single=[summarize([r],'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'macro_overlap':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normals)}
    assert len(source['fit_names'])==80
    variants={}; members={'first40':source['fit_names'][:40],'last40':source['fit_names'][40:]}
    assert not set(members['first40']) & set(members['last40'])
    for mode,names in members.items():
        print(f'{mode}: rebuilding from frozen 40-normal bank',flush=True)
        assert not set(names) & set(source['calibration_names'])
        bank=np.stack([features['train01',name] for name in names])
        model=mean_model(bank,np.array(source['selected_channels']))
        local_cal=[]; position_cal=[]
        for index,name in enumerate(source['calibration_names'],1):
            query=features['train01',name]
            local_cal.append(local_distance(query,bank,k=3,radius=1))
            position_cal.append(unweighted_score(query,model))
            if index%10==0: print(f'{mode}: calibration {index}/30',flush=True)
        local_scale,_=normal_scale(local_cal); position_scale,_=normal_scale(position_cal)
        maxima=[float(fuse(l,i,local_scale,position_scale).max()) for l,i in zip(local_cal,position_cal)]
        threshold=float(np.percentile(maxima,95))
        cases=[]; added=[]
        for index,(r,pool,baseline) in enumerate(traces,1):
            query=features['val01',r['image']]
            score=fuse(local_distance(query,bank,k=3,radius=1),unweighted_score(query,model),local_scale,position_scale)
            qualified=[b for b in pool if qualifies(b,region_score(b,score,r['trace']['width'],r['trace']['height']),threshold)]
            selected=anchored_selection(baseline,qualified,score,r['trace']['width'],r['trace']['height'],region_score)
            assert len(selected)==len(baseline) and (not selected or selected[0]==baseline[0])
            from tools.audit_cnn_eligibility import support_facts
            added.append({'image':r['image'],'added_selected_count':sum(not support_facts(b)['allowed'] for b in selected)})
            x,y,_,_=r['bounds']
            cases.append({'image':r['image'],'targets':r['targets'],'candidates':[
                {**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in selected]})
            if index%10==0: print(f'{mode}: validation {index}/30',flush=True)
        variants[mode]={'fit_names':names,'calibration_names':source['calibration_names'],'threshold':threshold,
                        'local_scale':local_scale,'position_scale':position_scale,'normal_calibration_maxima':maxima,
                        'overall':metrics(cases),'by_kind':{k:metrics([r for r in cases if r['image'].startswith(k+'_')])
                        for k in ('damaged','disconnected','misrouted','normal')},'cases':cases,'added':added}
    a.output.mkdir(parents=True)
    save(a.output/'report.json',{'protocol':'fixed_first_last40_normal_sensitivity_v1','variants':variants,'formal_path_changed':False,
         'source_report_sha256':hashlib.sha256((a.evidence/'original/report.json').read_bytes()).hexdigest(),
         'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'qualification_sha256':hashlib.sha256((a.repo/'tools/probe_cnn_qualified_support.py').read_bytes()).hexdigest(),
         'warning':'Preset ordinal halves, not verified independent scene groups. Both retain the same repeatedly examined val01 and calibration normals. Sensitivity evidence only; not external generalization or field accuracy.'})
    print(json.dumps({m:{'overall':d['overall'],'by_kind':d['by_kind'],'threshold':d['threshold']} for m,d in variants.items()},indent=2))


if __name__=='__main__': main()
