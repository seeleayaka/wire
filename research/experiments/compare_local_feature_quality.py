"""Post-hoc feature diagnostics at all annotated regions; never creates candidates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('E:/PythonProject10'))
    p.add_argument('--fault-cache',type=Path,required=True)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.repo))
    from tools.probe_local_normal_bank import region_score
    from tools.evaluate_mendeley_local_refinement import summarize
    records=[json.loads(f.read_text(encoding='utf-8')) for f in sorted(a.fault_cache.glob('*.json')) if f.name!='comparison.json']
    assert len(records)==15 and all(r['split']=='val01' for r in records)
    results={}
    for name,directory in (('coarse',a.coarse),('fine',a.fine)):
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        assert report['split']=='val01'
        by_image=[]
        for record in records:
            with np.load(directory/(Path(record['image']).stem+'_maps.npz')) as data:
                score=data['normal_bank']
            left,top,right,bottom=record['bounds']
            values=[]
            outside=0
            for x0,y0,x1,y1 in record['targets']:
                box={'left':max(0,x0-left),'top':max(0,y0-top),
                     'right':min(right-left,x1-left),'bottom':min(bottom-top,y1-top)}
                if box['right']<=box['left'] or box['bottom']<=box['top']:
                    outside+=1
                    continue
                values.append(region_score(box,score,right-left,bottom-top))
            by_image.append({'image':record['image'],'values':values,'outside_roi':outside,
                             'median':float(np.median(values)) if values else None})
        normal_p95=[r['bank_p95'] for r in report['maps'] if r['image'].startswith('normal_')]
        macro={}
        for mode,variant in report['results'].items():
            faults=[c for c in variant['cases'] if not c['image'].startswith('normal_')]
            per_image=[summarize([c],'candidates') for c in faults]
            macro[mode]={'image_macro_overlap_fraction':float(np.mean([m['target_boxes_with_overlap']/m['target_boxes'] for m in per_image])),
                         'image_macro_best_iou':float(np.mean([m['mean_best_target_iou'] for m in per_image]))}
        results[name]={'grid':report['grid'],'normal_image_p95_median':float(np.median(normal_p95)),
            'candidate_macro':macro,
            'by_kind':{kind:{'target_score_median':float(np.median([v for r in by_image if r['image'].startswith(kind+'_') for v in r['values']])),
                'image_median_score_median':float(np.median([r['median'] for r in by_image if r['image'].startswith(kind+'_')]))} for kind in ('damaged','disconnected','misrouted')},
            'by_image':by_image}
    output={'warning':'Descriptive annotation-region feature response, not fault accuracy. Fragmented targets and correlated photos are not independent.', 'results':results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='by_image'} for k,v in results.items()},indent=2))

if __name__=='__main__':
    main()
