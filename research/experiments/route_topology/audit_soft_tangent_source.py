"""Separate scalar multiscale/soft scoring/reciprocal graph audit."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_reference_color_paths_source import exact_regions

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/soft_tangent_source_20261008'


def scalar_angle(u,v):
    return math.degrees(math.atan2(abs(u[0]*v[1]-u[1]*v[0]),u[0]*v[0]+u[1]*v[1]))


def audit_family(family,raw,regions):
    independent,n=ndimage.label(raw,np.ones((3,3),int));_,labels=cv2.connectedComponents(raw.astype('uint8'),connectivity=8)
    for c in family['native_components']:
        selection=labels==c['component'];point=np.argwhere(selection)[0]
        assert np.array_equal(selection,independent==independent[tuple(point)])
        assert int(selection.sum())==c['pixels']
    assert n==len(family['native_components'])
    distance=ndimage.distance_transform_edt(np.pad(raw,1))[1:-1,1:-1]
    ends={e['endpoint_id']:e for e in family['endpoints']}
    for e in ends.values():
        pts=e['support_points_xy'];directions=[]
        for length in [6,9,13]:
            if len(pts)<length:continue
            tail=pts[length//2:length];x=pts[0][0]-sum(q[0] for q in tail)/len(tail);y=pts[0][1]-sum(q[1] for q in tail)/len(tail)
            norm=math.hypot(x,y);directions.append((x/norm,y/norm))
        mean=[sum(v[i] for v in directions)/len(directions) for i in [0,1]];norm=math.hypot(*mean)
        reliability=max(.25,min(1.,norm*len(directions)/3))
        assert np.allclose(e['outward_unit'],[v/norm for v in mean])
        assert abs(e['direction_reliability']-reliability)<1e-8
        width=2*np.median([distance[int(q[1]),int(q[0])] for q in pts[2:]])
        assert abs(e['width']-width)<1e-4
        assert all(raw[int(q[1]),int(q[0])] for q in pts)
    rankings={i:[] for i in ends};adj={c:set() for c in range(1,n+1)}
    for i,edge in enumerate(family['edges']):
        a,b=[ends[k] for k in edge['endpoint_ids']];p=a['point_xy'];q=b['point_xy']
        length=math.dist(p,q);d=[(q[j]-p[j])/length for j in [0,1]]
        angles=[scalar_angle(a['outward_unit'],d),scalar_angle(b['outward_unit'],[-v for v in d])]
        ratio=max(a['width'],b['width'])/min(a['width'],b['width']);scaled=length/((a['width']+b['width'])/2)
        reliability=min(a['direction_reliability'],b['direction_reliability']);possibilities=[]
        for handle in [.2,1/3,.5]:
            c1=[p[j]+a['outward_unit'][j]*length*handle for j in [0,1]]
            c2=[q[j]+b['outward_unit'][j]*length*handle for j in [0,1]]
            curve=[[(1-t)**3*p[j]+3*(1-t)**2*t*c1[j]+3*(1-t)*t*t*c2[j]+t**3*q[j] for j in [0,1]] for t in [i/32 for i in range(33)]]
            steps=[[v[j]-u[j] for j in [0,1]] for u,v in zip(curve,curve[1:])]
            if any(math.hypot(*s)<1e-9 or sum(s[j]*d[j] for j in [0,1])<=0 for s in steps):continue
            turn=sum(scalar_angle(u,v) for u,v in zip(steps,steps[1:]))
            if turn>165:continue
            wd=.45*reliability
            score=(wd*min(1.,sum(angles)/180)+.25*min(1.,scaled/24)+.15*min(1.,math.log(ratio)/math.log(4))+.15*min(1.,turn/150))/(wd+.55)+.1*(1-reliability)
            possibilities.append((score,handle,turn,curve))
        best=min(possibilities,key=lambda r:(r[0],r[1]))
        assert abs(best[0]-edge['score'])<1e-6 and best[1]==edge['handle']
        assert np.allclose(best[3],edge['virtual_curve_xy']) and len(possibilities)==edge['curve_variants']
        assert np.allclose(angles,edge['angles_deg'],atol=1e-6)
        assert max(angles)<=100 and scaled<=24 and ratio<=4
        assert edge['observed_gap_pixels']==0 and not edge['physical_identity_confirmed'] and not edge['score_is_probability']
        for endpoint in edge['endpoint_ids']:rankings[endpoint].append((edge['score'],i))
    for i,edge in enumerate(family['edges']):
        accept=True
        for endpoint in edge['endpoint_ids']:
            ranking=sorted(rankings[endpoint]);accept &= ranking[0][1]==i and ranking[0][0]<=.5 and (len(ranking)==1 or ranking[1][0]-ranking[0][0]>=.12)
        assert accept==(edge['state']=='fragment_pair_candidate')
        if accept:
            a,b=edge['components'];adj[a].add(b);adj[b].add(a)
    expected=[];seen=set()
    for start in adj:
        if start in seen:continue
        todo=[start];group=set()
        while todo:
            c=todo.pop()
            if c in group:continue
            group.add(c);todo.extend(adj[c]-group)
        seen|=group;selection=np.isin(labels,list(group));hits={name:int((selection&r).sum()) for name,r in regions.items()}
        if all(v>=8 for v in hits.values()):expected.append((sorted(group),hits,len(group)>1))
    assert expected==[(p['components'],p['anchor_hits'],p['contains_inferred_gap']) for p in family['anchor_path_candidates']]
    return len(ends),len(family['edges'])


def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(protocol['pins'])
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'));before=source_pins();assert before==protocol['mainline_pins']
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'));poses={r['id']:r for r in prep['cases']}
    scope_path=json.loads((base/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']
    scope=json.loads(Path(scope_path).read_text(encoding='utf-8'));ends=pairs=0
    for row in report['cases']:
        source=row['source_binding'];assert digest(source['path'])==source['image_sha256']
        rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(row['crop_box_xyxy']))
        regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors'])
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10;colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
        for rec in row['records']:
            assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
            mask=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128
            for f in rec['families']:
                raw=mask&colored&np.isin(bins,f['hue_bins']);path=OUT/(row['id']+'_'+rec['record_id']+'_family_'+str(f['hue_bins'][0])+'.png')
                assert np.array_equal(raw,np.asarray(Image.open(path).convert('L'))>0)
                n,m=audit_family(f,raw,regions);ends+=n;pairs+=m
            assert rec['soft_two_family_path_candidate']==all(f['anchor_path_candidates'] for f in rec['families'])
        assert row['soft_path']==any(r['soft_two_family_path_candidate'] for r in row['records'])
    gain=[r['id'] for r in report['cases'] if r['original_socket_state']=='mating_body_visible' and not r['old_hard_path'] and r['soft_path']]
    assert gain==report['gains']==['reference'];assert not report['source_gate_passed']
    verify(protocol['pins']);assert source_pins()==before
    result=dict(status='PASS',cases=5,endpoints=ends,pairs=pairs,independent_scalar_scores_and_multiscale=True,
       independent_graph_reachability=True,source_gain_is_inferred_only=True,GT_accuracy_not_computed=True,
       reused_pose_and_skeleton_not_second_model=True,mainline_unchanged=True,
       pins={str(p):digest(p) for p in [Path(__file__),OUT/'report.json',OUT/'protocol.json']})
    save(OUT/'audit_report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='pins'}))


if __name__=='__main__':main()
