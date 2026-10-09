"""Explain missing paths and rejected pairings without changing acceptance."""
import json
import math
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_reference_color_paths_source import exact_regions

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/soft_tangent_source_20261008'
OUT=ROOT/'artifacts/soft_tangent_diagnosis_20261008'


def main():
    protocol=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'));verify(protocol['pins'])
    before=source_pins();assert before==protocol['mainline_pins']
    report=json.loads((BASE/'report.json').read_text(encoding='utf-8'))
    control=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    poses={r['id']:r for r in json.loads((control/'preparation_report.json').read_text(encoding='utf-8'))['cases']}
    scope_path=json.loads((control/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']
    scope=json.loads(Path(scope_path).read_text(encoding='utf-8'));OUT.mkdir(exist_ok=False);rows=[]
    for row in report['cases']:
        rgb=np.asarray(Image.open(row['source_binding']['path']).convert('RGB').crop(row['crop_box_xyxy']))
        regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors'])
        results=[]
        for rec in row['records']:
            for family in rec['families']:
                name=row['id']+'_'+rec['record_id']+'_family_'+str(family['hue_bins'][0])+'.png'
                raw=np.asarray(Image.open(BASE/name).convert('L'))>0
                _,labels=cv2.connectedComponents(raw.astype('uint8'),connectivity=8)
                ends={e['endpoint_id']:e for e in family['endpoints']};rankings={e:[] for e in ends}
                for edge in family['edges']:
                    for e in edge['endpoint_ids']:rankings[e].append(edge)
                failures=Counter();rejected=[]
                for e,choices in rankings.items():
                    choices.sort(key=lambda x:x['score'])
                    if not choices:why='no_search_admissible_pair'
                    elif choices[0]['score']>.5:why='best_score_above_limit'
                    elif len(choices)>1 and choices[1]['score']-choices[0]['score']<.12:why='competing_scores'
                    elif choices[0]['state']!='fragment_pair_candidate':why='other_end_not_reciprocal'
                    else:why='accepted'
                    failures[why]+=1
                    rejected.append(dict(endpoint_id=e,point_xy=ends[e]['point_xy'],reason=why,
                         reliability=ends[e]['direction_reliability'],width=ends[e]['width'],
                         alternatives=[dict(endpoints=c['endpoint_ids'],score=c['score'],state=c['state']) for c in choices[:3]]))
                component_support=[]
                for c in family['native_components']:
                    hits={k:int(((labels==c['component'])&r).sum()) for k,r in regions.items()}
                    if any(hits.values()):component_support.append(dict(component=c['component'],pixels=c['pixels'],anchor_hits=hits,
                         endpoints=[e['endpoint_id'] for e in ends.values() if e['component']==c['component']]))
                results.append(dict(hue_bins=family['hue_bins'],qualified_endpoints=len(ends),
                     endpoint_dispositions=dict(failures),endpoint_diagnostics=rejected,anchor_components=component_support,
                     total_raw_anchor_hits={k:int((raw&r).sum()) for k,r in regions.items()},
                     weakest_direction_reliability=min((e['direction_reliability'] for e in ends.values()),default=None),
                     path_candidates=len(family['anchor_path_candidates'])))
        rows.append(dict(id=row['id'],families=results))
    verify(protocol['pins']);assert source_pins()==before
    result=dict(status='complete',cases=rows,acceptance_unchanged=True,mainline_unchanged=True,
      pins={str(p):digest(p) for p in [Path(__file__),BASE/'report.json',BASE/'protocol.json']},
      hypotheses_not_physical_GT=True,new_confirmed_connections=0)
    save(OUT/'report.json',result)
    print(json.dumps([dict(id=r['id'],families=[{k:v for k,v in f.items() if k not in ['endpoint_diagnostics','anchor_components']} for f in r['families']]) for r in rows]))
    for row in rows:
        if row['id']=='source_visible_01':print(json.dumps(row))


if __name__=='__main__':main()
