"""Frozen val01 parent-preserving hint probe, target labels used only for scoring."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('E:/PythonProject10')
sys.path.insert(0,str(ROOT))
from audit_cnn_precision_ceiling import reconstruct_groups
from inspection_agent.focus_hint import select_focus_hint
from tools.merge_audit import setup
from tools.evaluate_mendeley_local_refinement import summarize,iou


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('use fresh output')
    _,tiled=setup(ROOT)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    source=ROOT/'output/mendeley_cnn_qualified_support_20260929/report.json'
    frozen=load(source); existing={r['image']:r for r in frozen['results']['cnn_calibrated_support']['cases']}
    key=lambda b:tuple(b[k] for k in ('left','top','right','bottom'))
    cases=[];additional_precise=0;additional_01=0
    sha=hashlib.sha256((ROOT/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    for directory in ('mendeley_merge_audit_20260928','mendeley_normal_evidence_20260928'):
        for p in sorted((ROOT/'output'/directory).glob('*.json')):
            if p.name=='comparison.json':continue
            r=load(p); assert r['split']=='val01' and r['source_sha256']==sha
            t=r['trace'];x,y,_,_=r['bounds']
            groups=reconstruct_groups(t['raw'],tiled)
            group_map={key(tiled._merge_candidates(g)[0]):g for g in groups}
            with np.load(ROOT/'output/mendeley_cnn_heat_evidence_20260929/metric'/(p.stem+'_maps.npz'),allow_pickle=False) as data:score=data['fusion']
            parents=copy.deepcopy(existing[r['image']]['candidates']);hints=[];hierarchy=[]
            for index,parent in enumerate(parents):
                local={**parent,'left':parent['left']-x,'right':parent['right']-x,'top':parent['top']-y,'bottom':parent['bottom']-y}
                hint=select_focus_hint(local,group_map[key(local)],score,t['width'],t['height'],frozen['threshold'])
                if hint:
                    b=hint['box'];hint['box']={**b,'left':b['left']+x,'right':b['right']+x,'top':b['top']+y,'bottom':b['bottom']+y}
                    hint['parent_index']=index;hints.append(hint['box'])
                hierarchy.append({'parent':parent,'focus_hint':hint})
            assert parents==existing[r['image']]['candidates']
            for target in r['targets']:
                old=max((iou(b,target) for b in parents),default=0);new=max((iou(b,target) for b in hints),default=0)
                additional_precise+=old<.5<=new;additional_01+=old<.1<=new
            cases.append({'image':r['image'],'targets':r['targets'],'candidates':parents,'hints':hints,'hierarchy':hierarchy})
    assert len(cases)==30
    faults=[r for r in cases if r['targets']];normals=[r for r in cases if not r['targets']]
    baseline=summarize(faults,'candidates');assert baseline==frozen['results']['cnn_calibrated_support']['overall']['faults']
    metrics={'parents':baseline,'hint_only':summarize(faults,'hints'),
             'additional_precise_source_fragments':int(additional_precise),'additional_iou_01_source_fragments':int(additional_01),
             'normal_parent_count':sum(len(r['candidates']) for r in normals),'normal_hint_count':sum(len(r['hints']) for r in normals),
             'fault_images_with_hints':sum(bool(r['hints']) for r in faults),'normal_images_with_hints':sum(bool(r['hints']) for r in normals)}
    report={'split':'val01','policy':'one_hint_per_parent_nonedge_half_area_distinct_tile_iou25_support_CNNabove_frozen_normalmax95_rank_support_then_CNN',
            'threshold':frozen['threshold'],'parent_geometry_and_count_exact':True,'metrics':metrics,'cases':cases,
            'source_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'warning':'Hints are extra manual-review regions, not fault verdicts; parent+hint union metrics intentionally not advertised as accuracy. Human benefit untested; no test01 or default changes.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:json.dump(report,file,indent=2)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
