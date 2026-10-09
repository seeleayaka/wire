"""Scalar mapped covariance and equality-based footprint grouping audit.

Shared HSV/appearance and registered region primitive, not independent models.
"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from run_wire_local_shape_trial import prepare,ROOT,END,APPEAR,OUT as SHAPE_OUT
from run_endpoint_neighborhood_competition import OUT as COMP_OUT
from run_wire_appearance_trial import region_masks
from reference_wire_appearance_v3 import assess
from run_review import verified_run
from run_prompt_contrast import save,digest,verify
from bundle_runtime_pins import source_pins


def scalar_shape(mask,crop,H,stored):
    ys,xs=np.nonzero(mask);assert stored['points']==len(xs)
    if len(xs)<16:
        assert stored['state']=='insufficient_local_shape';return
    xy=[]
    for x,y in zip(xs,ys):
        x=float(x+crop[0]);y=float(y+crop[1])
        d=float(H[2][0]*x+H[2][1]*y+H[2][2]);assert abs(d)>=1e-9
        xy.append(((H[0][0]*x+H[0][1]*y+H[0][2])/d,(H[1][0]*x+H[1][1]*y+H[1][2])/d))
    mx=math.fsum(p[0] for p in xy)/len(xy);my=math.fsum(p[1] for p in xy)/len(xy)
    xx=math.fsum((x-mx)**2 for x,y in xy)/len(xy)
    yy=math.fsum((y-my)**2 for x,y in xy)/len(xy)
    cross=math.fsum((x-mx)*(y-my) for x,y in xy)/len(xy)
    span=math.hypot(xx-yy,2*cross);major=(xx+yy+span)/2;minor=max(0.,(xx+yy-span)/2)
    ratio=major/max(minor,1e-9);angle=(math.degrees(math.atan2(2*cross,xx-yy))/2)%180.
    assert stored['state']=='usable_local_shape'
    assert np.allclose(stored['covariance'],[[xx,cross],[cross,yy]],rtol=1e-8,atol=1e-7)
    assert np.allclose(stored['centroid_xy'],[mx,my],rtol=1e-10,atol=1e-7)
    assert math.isclose(stored['covariance_ratio'],ratio,rel_tol=1e-7,abs_tol=1e-7)
    difference=abs(stored['axis_degrees']-angle)%180.
    assert min(difference,180-difference)<1e-6
    assert stored['axis_usable']==(ratio>=2.)


def scalar_compare(reference,observed,stored):
    assert not stored['physical_identity_confirmed']
    if reference['state']!='usable_local_shape' or observed['state']!='usable_local_shape':
        assert stored['axis_state']==stored['elongation_state']=='insufficient_local_shape';return
    factor=max(reference['elongation'],observed['elongation'])/min(reference['elongation'],observed['elongation'])
    assert stored['elongation_state']==('local_elongation_reference_consistent' if factor<=4 else 'local_elongation_differs')
    if not reference['axis_usable'] or not observed['axis_usable']:
        assert stored['axis_state']=='insufficient_local_axis';return
    delta=abs(reference['axis_degrees']-observed['axis_degrees']);delta=min(delta,180.-delta)
    assert math.isclose(stored['axis_difference_degrees'],delta,abs_tol=1e-8)
    assert stored['axis_state']==('local_axis_reference_consistent' if delta<=45 else 'local_axis_differs')


def main():
    before=source_pins();protocol,prior,_,prepared,scope,_=prepare()
    shape_protocol=json.loads((SHAPE_OUT/'protocol.json').read_text(encoding='utf-8'))
    comp_protocol=json.loads((COMP_OUT/'protocol.json').read_text(encoding='utf-8'))
    verify(shape_protocol['pins']);verify(comp_protocol['pins'])
    shape_report=json.loads((SHAPE_OUT/'report.json').read_text(encoding='utf-8'))
    comp_report=json.loads((COMP_OUT/'report.json').read_text(encoding='utf-8'))
    shape_cases={r['id']:r for r in shape_report['cases']};comp_cases={r['id']:r for r in comp_report['cases']}
    assert comp_report['old_candidate_support_preserved']==sum(r['decision']=='reference_endpoint_pair_candidate' for r in prior['cases'])
    ref=prepared['reference'];crop=ref['crop_context']['crop_box_xyxy']
    rgb=np.asarray(Image.open(ref['original_source']['path']).convert('RGB').crop(crop))
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    inv=verified_run(base/'reference/cable_plus_reference_anatomy_box',ref['crop_context']['source']['path'])
    ref_masks=[np.asarray(Image.open(p).convert('L'))>0 for p,s in zip(inv['paths'],inv['scores']) if s>=.75]
    assert len(ref_masks)==1
    regions,scales=region_masks(rgb.shape[:2],crop,scope,ref['anchors'])
    for identity,region in regions.items():
        _,pixels=assess(rgb,ref_masks[0]&region,protocol['reference_profiles'][identity],scales[identity])
        H=next(p for p in ref['anchors'] if p['id']==identity)['inspection_to_reference_local']
        scalar_shape(pixels,crop,H,shape_protocol['reference_shapes'][identity])
    endpoint_count=0;native_records=0
    for row in prior['cases']:
        assert shape_cases[row['id']]['old_decision']==comp_cases[row['id']]['old_decision']==row['decision']
        equality_groups={a['id']:[] for a in scope['anchors']}
        if row['native_run_directory']:
            crop=row['crop_box_xyxy'];case=prepared[row['id']]
            assert digest(row['source_path'])==row['source_binding']['image_sha256']
            rgb=np.asarray(Image.open(row['source_path']).convert('RGB').crop(crop))
            inv=verified_run(row['native_run_directory'],case['crop_context']['source']['path'])
            regions,scales=region_masks(rgb.shape[:2],crop,scope,case['anchors'])
            recorded={r['record_id']:r for r in shape_cases[row['id']]['records']}
            for path,score in zip(inv['paths'],inv['scores']):
                if score<.75:continue
                native_records+=1;raw=np.asarray(Image.open(path).convert('L'))>0
                for identity,region in regions.items():
                    color,pixels=assess(rgb,raw&region,protocol['reference_profiles'][identity],scales[identity])
                    data=recorded[path.stem]['endpoints'][identity]
                    H=next(p for p in case['anchors'] if p['id']==identity)['inspection_to_reference_local']
                    scalar_shape(pixels,crop,H,data['shape'])
                    scalar_compare(shape_protocol['reference_shapes'][identity],data['shape'],data['comparison'])
                    endpoint_count+=1
                    if color['state']=='reference_color_supported' and pixels.any():
                        groups=equality_groups[identity]
                        same=next((g for g in groups if np.array_equal(g['mask'],pixels)),None)
                        if same is None:groups.append(dict(mask=pixels.copy(),ids=[path.stem]))
                        else:same['ids'].append(path.stem)
        result=comp_cases[row['id']]['competition']
        for identity,groups in equality_groups.items():
            stored=result['endpoints'][identity]
            assert stored['distinct_footprints']==len(groups)
            expected=sorted(sorted(g['ids']) for g in groups)
            actual=sorted(g['native_record_ids'] for g in stored['footprints'])
            assert actual==expected
            assert stored['state']==('no_supported_local_footprint' if not groups else 'single_supported_local_footprint' if len(groups)==1 else 'competing_visible_local_footprints')
        assert result['review_needed']==any(len(g)>1 for g in equality_groups.values())
        assert not result['same_physical_wire_confirmed']
    assert source_pins()==before
    output=dict(status='PASS',cases=30,reference_descriptors=2,endpoint_descriptors=endpoint_count,
        high_score_native_records=native_records,scalar_coordinate_mapping_covariance_and_analytic_axis=True,
        footprint_groups_recomputed_by_pixel_equality=True,
        shared_primitives=['HSV/appearance selection','registered region mapping','native inventory verification'],
        not_an_independent_visual_model=True,physical_identity_gt_available=False,
        shape_report_sha256=digest(SHAPE_OUT/'report.json'),competition_report_sha256=digest(COMP_OUT/'report.json'))
    save(SHAPE_OUT/'audit_report.json',output);save(COMP_OUT/'audit_report.json',output)
    print(json.dumps(output))


if __name__=='__main__':main()
