"""Separate raw mask/component/score/reciprocity audit; no matcher calls."""
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

ROOT=Path(__file__).resolve().parents[2]


def audit_family(family,raw,regions,params):
    scipy_labels,count=ndimage.label(raw,structure=np.ones((3,3),int))
    # Label numbers are library-local, not stable semantic IDs. Verify partitions,
    # then map the independently flooded SciPy groups to recorded OpenCV IDs.
    cv_count,cv_labels=cv2.connectedComponents(raw.astype('uint8'),connectivity=8)
    assert cv_count-1==count
    labels=np.zeros_like(scipy_labels)
    for component in range(1,cv_count):
        first=np.argwhere(cv_labels==component)[0]
        independent_id=scipy_labels[tuple(first)]
        selection=scipy_labels==independent_id
        assert np.array_equal(selection,cv_labels==component)
        labels[selection]=component
    assert count==len(family['native_components'])
    for c in family['native_components']:assert int((labels==c['component']).sum())==c['pixels']
    distance=ndimage.distance_transform_edt(np.pad(raw,1))[1:-1,1:-1]
    ends={e['endpoint_id']:e for e in family['endpoints']}
    for e in ends.values():
        points=e['support_points_xy'];x,y=map(int,points[0]);assert raw[y,x]
        assert labels[y,x]==e['component']
        assert all(raw[int(q[1]),int(q[0])] and labels[int(q[1]),int(q[0])]==e['component'] for q in points)
        assert all(max(abs(a[0]-b[0]),abs(a[1]-b[1]))==1 for a,b in zip(points,points[1:]))
        width=2*np.median([distance[int(q[1]),int(q[0])] for q in points[2:]])
        assert abs(width-e['width'])<1e-4
    rankings={e:[] for e in ends};adj={c:set() for c in range(1,count+1)}
    for index,edge in enumerate(family['edges']):
        a,b=[ends[e] for e in edge['endpoint_ids']]
        delta=np.array(b['point_xy'])-a['point_xy'];length=float(np.linalg.norm(delta));unit=delta/length
        angles=[]
        for direction,target in [(a['outward_unit'],unit),(b['outward_unit'],-unit)]:
            cross=direction[0]*target[1]-direction[1]*target[0]
            angles.append(math.degrees(math.atan2(abs(cross),np.dot(direction,target))))
        ratio=max(a['width'],b['width'])/min(a['width'],b['width']);scaled=length/((a['width']+b['width'])/2)
        score=(max(angles)/params['max_angle_deg']+scaled/params['max_gap_widths']+math.log(ratio)/math.log(2))/3
        assert np.allclose(angles,edge['angles_deg'],atol=1e-6) and abs(score-edge['score'])<1e-6
        assert max(angles)<=35 and ratio<=2 and scaled<=12 and edge['accumulated_turn_deg']<=70
        curve=np.array(edge['virtual_curve_xy']);steps=np.diff(curve,axis=0)
        assert (steps@unit>0).all()
        assert edge['observed_gap_pixels']==0 and edge['physical_identity_confirmed'] is False
        for endpoint in edge['endpoint_ids']:rankings[endpoint].append((score,index))
    for index,edge in enumerate(family['edges']):
        good=True
        for endpoint in edge['endpoint_ids']:
            ranking=sorted(rankings[endpoint])
            good &= ranking[0][1]==index and (len(ranking)==1 or ranking[1][0]-ranking[0][0]>=params['margin'])
        assert (edge['state']=='fragment_pair_candidate')==good
        if good:
            a,b=edge['components'];adj[a].add(b);adj[b].add(a)
    observed=[];seen=set()
    for c in adj:
        if c in seen:continue
        todo=[c];group=set()
        while todo:
            q=todo.pop()
            if q in group:continue
            group.add(q);todo.extend(adj[q]-group)
        seen|=group
        hits={name:int((np.isin(labels,list(group))&region).sum()) for name,region in regions.items()}
        if all(v>=8 for v in hits.values()):observed.append((sorted(group),hits,len(group)>1))
    actual=[(p['components'],p['anchor_hits'],p['contains_inferred_gap']) for p in family['anchor_path_candidates']]
    assert observed==actual
    assert family['observed_pixels_added']==0 and family['electrical_continuity']=='not_assessed'
    return len(ends),len(family['edges'])


def main():
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'))
    original={r['id']:r for r in prep['cases']}
    prior=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    scope=json.loads(Path(prior['confirmed_scope_path']).read_text(encoding='utf-8'))
    before=source_pins();assert before==prior['mainline_pins']
    for variant in ['tangent_gap_source_20261008','tangent_gap_native_bound_20261008']:
        out=ROOT/'artifacts'/variant;protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        report=json.loads((out/'report.json').read_text(encoding='utf-8'));verify(protocol['pins'])
        endpoints=pairs=families=0
        for row in report['cases']:
            original_row=original[row['id']];source=row['source_binding']
            assert digest(source['path'])==source['image_sha256']
            rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(row['crop_box_xyxy']))
            regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,original_row['anchors'])
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10
            colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
            records=row.get('records',[dict(families=row.get('families',[]))])
            for rec in records:
                mask=np.ones(rgb.shape[:2],bool)
                if 'mask_path' in rec:
                    assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
                    mask=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128
                for family in rec['families']:
                    raw=colored&np.isin(bins,family['hue_bins'])&mask
                    suffix=(row['id']+'_'+rec['record_id'] if 'record_id' in rec else row['id'])+'_family_'+str(family['hue_bins'][0])+'.png'
                    assert np.array_equal(raw,np.asarray(Image.open(out/suffix).convert('L'))>0)
                    n,m=audit_family(family,raw,regions,protocol['params']);endpoints+=n;pairs+=m;families+=1
            assert row['physical_identity_confirmed'] is False and row['electrical_continuity']=='not_assessed'
        verify(protocol['pins']);assert source_pins()==before
        save(out/'audit_report.json',dict(status='PASS',source_cases=5,families=families,
            endpoint_descriptors=endpoints,geometry_pairs=pairs,raw_pixel_binding_verified=True,
            independent_scipy_components_and_widths=True,independent_scalar_angles_scores_and_reciprocity=True,
            graph_reachability_recomputed=True,skimage_skeleton_not_independently_reimplemented=True,
            no_physical_identity_GT=True,no_accuracy_claim=True,mainline_unchanged=True,
            pins={str(p):digest(p) for p in [Path(__file__),out/'report.json',out/'protocol.json']}))
        print(json.dumps(dict(variant=variant,status='PASS',families=families,endpoints=endpoints,pairs=pairs)))


if __name__=='__main__':main()
