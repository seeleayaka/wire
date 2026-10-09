"""Frozen reference local shape, all30 in shadow mode; original reports untouched."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image,ImageDraw
from run_wire_appearance_trial import region_masks,read
from reference_wire_appearance_v3 import assess
from reference_wire_local_shape import describe,compare,POLICY
from run_prompt_contrast import save,digest,verify
from run_review import verified_run
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2]
END=ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008'
APPEAR=ROOT/'artifacts/wire_appearance_reference_v3_20261008'
OUT=ROOT/'artifacts/wire_local_shape_shadow_20261008'


def prepare():
    protocol=read(APPEAR/'protocol.json');verify(protocol['pins'])
    endpoint_protocol=read(END/'protocol.json');verify(endpoint_protocol['pins'])
    appearance=read(APPEAR/'report.json');prior=read(END/'report.json')
    assert appearance['status']==prior['status']=='complete'
    assert len(prior['cases'])==30
    base=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    recent=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    prepared={r['id']:r for r in read(base/'preparation_report.json')['cases']}
    prepared.update({r['id']:r for r in read(recent/'preparation_report.json')['cases']})
    scope_path=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    scope=read(scope_path)
    ref=prepared['reference'];crop=ref['crop_context']['crop_box_xyxy']
    rgb=np.asarray(Image.open(ref['original_source']['path']).convert('RGB').crop(crop))
    inventory=verified_run(base/'reference/cable_plus_reference_anatomy_box',ref['crop_context']['source']['path'])
    masks=[np.asarray(Image.open(path).convert('L'))>0 for path,score in zip(inventory['paths'],inventory['scores']) if score>=.75]
    assert len(masks)==1
    regions,scales=region_masks(rgb.shape[:2],crop,scope,ref['anchors'])
    profiles=protocol['reference_profiles'];shapes={}
    for identity,region in regions.items():
        _,selected=assess(rgb,masks[0]&region,profiles[identity],scales[identity])
        H=next(p for p in ref['anchors'] if p['id']==identity)['inspection_to_reference_local']
        shapes[identity]=describe(selected,crop[:2],H)
    return protocol,prior,appearance,prepared,scope,shapes


def main():
    started=time.monotonic();protocol,prior,appearance,prepared,scope,shapes=prepare()
    before=source_pins();assert before==protocol['mainline_pins']
    files=[Path(__file__),Path(__file__).with_name('reference_wire_local_shape.py'),
           Path(__file__).with_name('test_reference_wire_local_shape.py'),END/'report.json',
           APPEAR/'report.json',APPEAR/'protocol.json',ROOT/'artifacts/WIRE_LOCAL_SHAPE_PROTOCOL_20261008.md']
    pins={str(p):digest(p) for p in files}
    OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,policy=POLICY,
        reference_shapes=shapes,reference_color_profiles=protocol['reference_profiles'],
        shape_is_not_tangent_or_identity=True,shared_registered_regions=True,fresh_SAM_calls=0))
    cases=[]
    for row in prior['cases']:
        records=[];selected_id=row['candidate_records'][0]['record_id'] if row['decision']=='reference_endpoint_pair_candidate' else None
        if row['native_run_directory']:
            crop=row['crop_box_xyxy'];source=row['source_path']
            assert digest(source)==row['source_binding']['image_sha256']
            rgb=np.asarray(Image.open(source).convert('RGB').crop(crop))
            case=prepared[row['id']];context=case['crop_context']
            assert np.array_equal(rgb,np.asarray(Image.open(context['source']['path']).convert('RGB')))
            inv=verified_run(row['native_run_directory'],context['source']['path'])
            regions,scales=region_masks(rgb.shape[:2],crop,scope,case['anchors'])
            display=rgb.copy();annotations=[]
            for path,score in zip(inv['paths'],inv['scores']):
                if score<.75:continue
                raw=np.asarray(Image.open(path).convert('L'))>0;endpoints={}
                for identity,region in regions.items():
                    color,selection=assess(rgb,raw&region,protocol['reference_profiles'][identity],scales[identity])
                    H=next(p for p in case['anchors'] if p['id']==identity)['inspection_to_reference_local']
                    shape=describe(selection,crop[:2],H);comparison=compare(shapes[identity],shape)
                    endpoints[identity]=dict(color=color,shape=shape,comparison=comparison)
                    if path.stem==selected_id:
                        flag=comparison['axis_state']=='local_axis_differs' or comparison['elongation_state']=='local_elongation_differs'
                        tint=np.array([240,80,20] if flag else [0,180,220])
                        display[selection]=(display[selection]*.4+tint*.6).astype('uint8')
                        angle=comparison.get('axis_difference_degrees')
                        annotations.append(identity+': axis '+('unknown' if angle is None else f'{angle:.1f} deg')+' / '+comparison['elongation_state'])
                records.append(dict(record_id=path.stem,score=float(score),selected_old_candidate=path.stem==selected_id,endpoints=endpoints))
            if selected_id:
                h,w=rgb.shape[:2];canvas=Image.new('RGB',(max(w,650),h+105),'white');canvas.paste(Image.fromarray(display),(0,0))
                draw=ImageDraw.Draw(canvas);draw.text((8,h+6),row['id']+' local geometry SHADOW, not wire identity',fill='black')
                for i,label in enumerate(annotations):draw.text((8,h+26+i*19),label,fill='black')
                draw.text((8,h+80),'orange: local shape differs; old candidate remains unchanged',fill='black')
                canvas.save(OUT/(row['id']+'.png'))
        chosen=next((r for r in records if r['selected_old_candidate']),None)
        axis_ok=bool(chosen and len(chosen['endpoints'])==2 and all(e['comparison']['axis_state']=='local_axis_reference_consistent' for e in chosen['endpoints'].values()))
        elongation_ok=bool(chosen and len(chosen['endpoints'])==2 and all(e['comparison']['elongation_state']=='local_elongation_reference_consistent' for e in chosen['endpoints'].values()))
        cases.append(dict(id=row['id'],old_decision=row['decision'],records=records,
            old_candidate_axis_supported=axis_ok,old_candidate_elongation_supported=elongation_ok,
            old_candidate_combined_shape_supported=axis_ok and elongation_ok,
            old_decision_unchanged=True,physical_identity_confirmed=False))
    old=[r for r in cases if r['old_decision']=='reference_endpoint_pair_candidate']
    summary=dict(status='complete',cases=cases,total_cases=len(cases),old_candidates=len(old),
        axis_supported=sum(r['old_candidate_axis_supported'] for r in old),
        elongation_supported=sum(r['old_candidate_elongation_supported'] for r in old),
        combined_shape_supported=sum(r['old_candidate_combined_shape_supported'] for r in old),
        candidate_support_losses_if_combined_gate_used=[r['id'] for r in old if not r['old_candidate_combined_shape_supported']],
        physical_identity_confirmed=0,electrical_connections_confirmed=0,fresh_SAM_calls=0,
        original_decisions_unchanged=True,not_field_accuracy=True,not_deployed=True,seconds=time.monotonic()-started)
    verify(pins);assert source_pins()==before
    save(OUT/'report.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))


if __name__=='__main__':main()
