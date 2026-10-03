"""Two fixed validation probes. Geometry selection never reads target labels."""
import copy
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from audit_cnn_precision_ceiling import reconstruct_groups
ROOT=Path('E:/PythonProject10')
sys.path.insert(0,str(ROOT))
from tools.merge_audit import setup
from tools.probe_local_normal_bank import region_score
from tools.evaluate_mendeley_local_refinement import summarize,box_area


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('use fresh output')
    _,tiled=setup(ROOT)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    frozen=load(ROOT/'output/mendeley_cnn_qualified_support_20260929/report.json')
    existing={c['image']:c for c in frozen['results']['cnn_calibrated_support']['cases']}
    key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    cases={m:[] for m in ('baseline','member_cnn','member_consensus_cnn')}
    changes={m:0 for m in cases}
    for directory in ('mendeley_merge_audit_20260928','mendeley_normal_evidence_20260928'):
        for p in sorted((ROOT/'output'/directory).glob('*.json')):
            if p.name=='comparison.json':continue
            r=load(p);assert r['split']=='val01'
            t=r['trace'];x,y,_,_=r['bounds']
            parents=[{**b,'left':b['left']-x,'right':b['right']-x,'top':b['top']-y,'bottom':b['bottom']-y}
                     for b in existing[r['image']]['candidates']]
            groups=reconstruct_groups(t['raw'],tiled)
            members={key(tiled._merge_candidates(g)[0]):g for g in groups}
            anchor=key(t['local_candidates'][0]) if t['local_candidates'] else None
            with np.load(ROOT/'output/mendeley_cnn_heat_evidence_20260929/metric'/(p.stem+'_maps.npz'),allow_pickle=False) as data:
                score=data['fusion']
            chosen={'baseline':parents}
            for mode in ('member_cnn','member_consensus_cnn'):
                rows=[]
                for parent in parents:
                    candidates=[b for b in members[key(parent)] if b.get('source') in ('tile','refinement')
                                and not b.get('touches_tile_edge') and box_area(b)>0]
                    if key(parent)==anchor or not candidates:
                        rows.append(copy.deepcopy(parent));continue
                    def rank(b):
                        novelty=region_score(b,score,t['width'],t['height'])
                        if mode=='member_cnn':return (novelty,float(b['difference_score']))
                        corroboration=sum(tiled._iou(b,c)>=.25 for c in members[key(parent)]
                                          if set(b['source_tiles']).isdisjoint(c['source_tiles']))
                        return (corroboration,novelty,float(b['difference_score']))
                    best=max(candidates,key=rank)
                    rows.append(copy.deepcopy(best));changes[mode]+=key(best)!=key(parent)
                assert len(rows)==len(parents)
                chosen[mode]=rows
            for mode,rows in chosen.items():
                absolute=[{**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y} for b in rows]
                cases[mode].append({'image':r['image'],'targets':r['targets'],'candidates':absolute})
    def metrics(rows):
        faults=[r for r in rows if r['targets']];normals=[r for r in rows if not r['targets']]
        return {'faults':summarize(faults,'candidates'),'normal_candidate_count':sum(len(r['candidates']) for r in normals),
                'by_kind':{k:summarize([r for r in faults if r['image'].startswith(k+'_')],'candidates') for k in ('damaged','disconnected','misrouted')}}
    results={m:metrics(rows) for m,rows in cases.items()}
    assert results['baseline']['faults']==frozen['results']['cnn_calibrated_support']['overall']['faults']
    output=args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as file:
        json.dump({'split':'val01','results':results,'changed_boxes':changes,'cases':cases,
             'policy':'Keep original anchor, replace each other selected parent with one non-edge tile/refinement member; rank by CNN p95 or different-tile IoU>=.25 support then CNN p95. Fixed policies, no thresholds tuned.',
             'warning':'Validation probes only; no default path change, no test01 read. Parent coverage can be lost; equal normal count does not mean equal normal burden.'},file,indent=2)
    print(json.dumps({'results':{m:v['faults'] for m,v in results.items()},'changes':changes},indent=2))


if __name__=='__main__':main()
