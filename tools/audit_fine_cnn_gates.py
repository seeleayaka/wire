"""Explain fixed fine gate outcomes, without changing thresholds or reading targets."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('E:/PythonProject10');sys.path.insert(0,str(ROOT))
from tools.merge_audit import setup
from tools.audit_cnn_precision_ceiling import reconstruct_groups
from tools.probe_local_normal_bank import region_score
from tools.probe_cnn_qualified_support import qualifies
from tools.audit_cnn_eligibility import support_facts
from inspection_agent.focus_hint import box_area,intersection_over_union


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('fresh output')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=load(args.report);assert report['split']=='val01'
    _,tiled=setup(ROOT);threshold=report['threshold'];cases={r['image']:r for r in report['cases']}
    key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    counts={g:dict(pre_cnn_supported_hint_members=0,above_threshold_supported_members=0,extra_qualified_selected=0,extra_qualified_pool=0,max_pre_cnn_member_score=0.) for g in ('fault','normal')}
    bounds=load(ROOT/'output/mendeley_cnn_heat_evidence_20260929/original/report.json')['bounds'];x,y,_,_=bounds
    for directory in ('mendeley_merge_audit_20260928','mendeley_normal_evidence_20260928'):
        for p in sorted((ROOT/'output'/directory).glob('*.json')):
            if p.name=='comparison.json':continue
            r=load(p);assert r['split']=='val01';t=r['trace'];kind='normal' if r['image'].startswith('normal_') else 'fault'
            with np.load(args.report.parent/(p.stem+'_maps.npz'),allow_pickle=False) as data:score=data['fusion']
            pool=tiled._merge_candidates(copy.deepcopy(t['raw']));tiled._annotate_roi_edges(pool,t['width'],t['height'])
            pool,_=tiled._publish_candidates(pool,[])
            extra={key(b) for b in pool if not support_facts(b)['allowed'] and qualifies(b,region_score(b,score,t['width'],t['height']),threshold)}
            counts[kind]['extra_qualified_pool']+=len(extra)
            for b in cases[r['image']]['fine_selected']:
                local={**b,'left':b['left']-x,'right':b['right']-x,'top':b['top']-y,'bottom':b['bottom']-y}
                counts[kind]['extra_qualified_selected']+=key(local) in extra
            groups=reconstruct_groups(t['raw'],tiled);members={key(tiled._merge_candidates(g)[0]):g for g in groups}
            for parent in cases[r['image']]['parents']:
                local={**parent,'left':parent['left']-x,'right':parent['right']-x,'top':parent['top']-y,'bottom':parent['bottom']-y}
                group=members[key(local)]
                for child in group:
                    if child.get('source') not in ('tile','refinement') or child.get('touches_tile_edge') or box_area(child)>.5*box_area(local):continue
                    if not any(set(child['source_tiles']).isdisjoint(other['source_tiles']) and intersection_over_union(child,other)>=.25 for other in group):continue
                    value=region_score(child,score,t['width'],t['height'])
                    counts[kind]['pre_cnn_supported_hint_members']+=1
                    counts[kind]['above_threshold_supported_members']+=value>threshold
                    counts[kind]['max_pre_cnn_member_score']=max(counts[kind]['max_pre_cnn_member_score'],value)
    with args.output.open('x',encoding='utf-8') as file:json.dump({'threshold':threshold,'counts':counts,'rule_changed':False,'warning':'Explanation only, not a threshold search or accuracy measure.'},file,indent=2)
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
