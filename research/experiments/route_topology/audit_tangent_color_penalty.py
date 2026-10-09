"""Independently verify color histograms, penalties and virtual graph paths."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_reference_color_paths_source import exact_regions

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/tangent_color_penalty_20261008'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(protocol['pins'])
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'artifacts/soft_tangent_source_20261008/report.json').read_text(encoding='utf-8'))
    old_by_id={r['id']:r for r in old['cases']};before=source_pins();assert before==protocol['mainline_pins']
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'));poses={r['id']:r for r in prep['cases']}
    scope=json.loads(Path(json.loads((base/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']).read_text(encoding='utf-8'))
    count=edges=0;coverage=[]
    for row in report['cases']:
        prior=old_by_id[row['id']];source=row['source_binding'];assert digest(source['path'])==source['image_sha256']
        rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(prior['crop_box_xyxy']))
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10
        regions=exact_regions(rgb.shape[:2],prior['crop_box_xyxy'],scope,poses[row['id']]['anchors'])
        colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
        for rec,oldrec in zip(row['records'],prior['records']):
            for f,oldf in zip(rec['families'],oldrec['families']):
                path=ROOT/'artifacts/soft_tangent_source_20261008'/(row['id']+'_'+rec['record_id']+'_family_'+str(f['hue_bins'][0])+'.png')
                raw=np.asarray(Image.open(path).convert('L'))>0;_,labels=cv2.connectedComponents(raw.astype('uint8'),connectivity=8)
                coverage.append(dict(id=row['id'],hue_bins=f['hue_bins'],raw_RGB_anchor_hits={k:int((colored&np.isin(bins,f['hue_bins'])&r).sum()) for k,r in regions.items()},SAM_bound_anchor_hits=f['raw_anchor_hits']))
                assert len(f['endpoints'])==len(oldf['endpoints'])
                for e,original in zip(f['endpoints'],oldf['endpoints']):
                    assert {k:v for k,v in e.items() if k!='color_histogram'}==original
                    radius=max(1,int(round(e['width']/2)));points=set();h,w=raw.shape
                    for x,y in e['support_points_xy']:
                        for yy in range(max(0,int(y)-radius),min(h,int(y)+radius+1)):
                            for xx in range(max(0,int(x)-radius),min(w,int(x)+radius+1)):
                                if raw[yy,xx] and labels[yy,xx]==e['component']:points.add((yy,xx))
                    hist=[0]*18
                    for y,x in points:hist[int(bins[y,x])]+=1
                    assert np.allclose(e['color_histogram'],[v/len(points) for v in hist]);count+=1
                ends={e['endpoint_id']:e for e in f['endpoints']};oldedges={tuple(e['endpoint_ids']):e for e in oldf['edges']};rank={e:[] for e in ends};adj={c['component']:set() for c in f['native_components']}
                for i,e in enumerate(f['edges']):
                    original=oldedges[tuple(e['endpoint_ids'])];a,b=[ends[k] for k in e['endpoint_ids']]
                    similarity=sum(math.sqrt(x*y) for x,y in zip(a['color_histogram'],b['color_histogram']));distance=math.sqrt(max(0.,1-similarity))
                    assert abs(distance-e['color_distance'])<1e-8
                    assert abs(e['score']-(original['score']+.25*distance))<1e-8 and e['score']>=original['score']
                    for endpoint in e['endpoint_ids']:rank[endpoint].append((e['score'],i))
                    edges+=1
                for i,e in enumerate(f['edges']):
                    good=True
                    for endpoint in e['endpoint_ids']:
                        ranking=sorted(rank[endpoint]);good &= ranking[0][1]==i and ranking[0][0]<=.5 and (len(ranking)==1 or ranking[1][0]-ranking[0][0]>=.12)
                    assert good==(e['state']=='fragment_pair_candidate')
                    if good:
                        a,b=e['components'];adj[a].add(b);adj[b].add(a)
                paths=[];seen=set()
                for c in adj:
                    if c in seen:continue
                    group=set();todo=[c]
                    while todo:
                        q=todo.pop()
                        if q in group:continue
                        group.add(q);todo.extend(adj[q]-group)
                    seen|=group;selected=np.isin(labels,list(group));hits={k:int((selected&r).sum()) for k,r in regions.items()}
                    if all(v>=8 for v in hits.values()):paths.append((sorted(group),hits,len(group)>1))
                assert paths==[(p['components'],p['anchor_hits'],p['contains_inferred_gap']) for p in f['anchor_path_candidates']]
    verify(protocol['pins']);assert source_pins()==before
    result=dict(status='PASS',source_cases=5,endpoint_color_profiles=count,pair_penalties=edges,
         independent_pixel_histograms_and_scalar_penalties=True,independent_reciprocity_and_graph=True,
         coverage_diagnosis=coverage,mainline_unchanged=True,not_physical_identity_GT=True,
         pins={str(p):digest(p) for p in [Path(__file__),OUT/'report.json',OUT/'protocol.json']})
    save(OUT/'audit_report.json',result);print(json.dumps(dict(status='PASS',profiles=count,edges=edges)))
    print(json.dumps([r for r in coverage if r['id']=='source_visible_01']))


if __name__=='__main__':main()
