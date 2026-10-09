"""Replay verified native SAM; local endpoint record association, not new SAM."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from core import image_binding
from run_prompt_contrast import save,digest,verify
from bundle_runtime_pins import source_pins
from run_review import verified_run
from visible_bundle_relation import observe_bundle
from endpoint_pair_candidate import compare_endpoint_pair

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/endpoint_pair_trial_v3_20261008'


def endpoint_quads(scope,case):
    result={}
    for anchor in scope['anchors']:
        pose=next(p for p in case['anchors'] if p['id']==anchor['id'])
        if not pose['localization_proposal_supported'] or not all(pose['gates'].values()):continue
        l,t,r,b=anchor['bbox_xyxy']
        points=np.array([[l,t,1],[r,t,1],[r,b,1],[l,b,1]],float)
        q=points@np.linalg.inv(np.array(pose['inspection_to_reference_local'])).T
        result[anchor['id']]=(q[:,:2]/q[:,2:]).tolist()
    return result


def render(scope,case,view,result,path):
    crop=case['crop_context']['crop_box_xyxy']
    image=Image.open(case['original_source']['path']).convert('RGB').crop(crop)
    rgb=np.asarray(image).copy()
    for record in view['_raw_masks']:
        mask=record['raw']>0
        if record['score']>=.75:
            rgb[mask]=(rgb[mask]*.45+np.array([0,200,255])*.55).astype('uint8')
    canvas=Image.new('RGB',(rgb.shape[1],rgb.shape[0]+90),'white')
    canvas.paste(Image.fromarray(rgb),(0,0))
    draw=ImageDraw.Draw(canvas)
    for identity,quad in endpoint_quads(scope,case).items():
        local=[(x-crop[0],y-crop[1]) for x,y in quad]
        draw.line(local+[local[0]],fill=(255,120,0),width=2)
        draw.text(local[0],identity,fill=(255,120,0))
    draw.text((8,rgb.shape[0]+8),result['decision'],fill='black')
    draw.text((8,rgb.shape[0]+30),'middle: '+result['middle_connection'],fill='black')
    draw.text((8,rgb.shape[0]+52),'same physical wire NOT confirmed',fill='black')
    canvas.save(path)


def main():
    old=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    p=json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    verify(p['pins'])
    assert source_pins()==p['mainline_pins']
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'))
    cases=json.loads((old/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    OUT.mkdir(exist_ok=False)
    pins={str(Path(__file__).resolve()):digest(__file__),
        str(ROOT/'experiments/route_topology/endpoint_pair_candidate.py'):digest(ROOT/'experiments/route_topology/endpoint_pair_candidate.py')}
    views={}
    for case in cases:
        crop=case['crop_context']['crop_box_xyxy']
        image=Image.open(case['original_source']['path']).convert('RGB')
        actual_binding=image_binding(case['original_source']['path'])
        assert actual_binding=={k:case['original_source'][k] for k in actual_binding}
        assert np.array_equal(np.asarray(image.crop(crop)),np.asarray(Image.open(case['crop_context']['source']['path']).convert('RGB')))
        run=old/case['id']/'cable_plus_reference_anatomy_box'
        inventory=verified_run(run,case['crop_context']['source']['path'])
        masks=[dict(raw=np.asarray(Image.open(path).convert('L')),score=score,
                    record_id=path.stem,recipe='cable_plus_reference_anatomy_box')
               for path,score in zip(inventory['paths'],inventory['scores'])]
        view=observe_bundle(scope,actual_binding,case['anchors'],masks,case['phenotype'],
            translation=tuple(crop[:2]),sam_inventory_verified=True)
        view['_raw_masks']=masks
        views[case['id']]=view
    results=[]
    for case in cases:
        result=compare_endpoint_pair(scope,views['reference'],views[case['id']])
        result.update(id=case['id'],image_binding=views[case['id']]['source_binding'],
            endpoint_source_quads=endpoint_quads(scope,case),
            input_provenance='replay_of_verified_original_native_SAM_and_prior_local_poses',fresh_SAM=False)
        render(scope,case,views[case['id']],result,OUT/(case['id']+'.png'))
        results.append(result)
    # Cross-check this result with the recent fresh SAM decoder inventories, not another observer.
    fresh=ROOT/'artifacts/sam_prompt_confidence_source_20261008'
    fp=json.loads((fresh/'protocol.json').read_text(encoding='utf-8'))
    verify(fp['pins'])
    fresh_views={}
    for case_id in ['reference','source_visible_01']:
        case=next(c for c in cases if c['id']==case_id)
        inventory=json.loads((fresh/case_id/'cable_box/inventory.json').read_text())
        native=fresh/case_id/'cable_box/native.npz'
        assert digest(native)==inventory['native_sha256']
        arrays=np.load(native,allow_pickle=False)
        masks=[dict(raw=mask,score=float(score),record_id='native_%03d'%i,recipe='cable_plus_reference_anatomy_box')
               for i,(mask,score) in enumerate(zip(arrays['masks'],arrays['scores'])) if score>.5]
        fresh_views[case_id]=observe_bundle(scope,image_binding(case['original_source']['path']),case['anchors'],masks,case['phenotype'],
            translation=tuple(case['crop_context']['crop_box_xyxy'][:2]),sam_inventory_verified=True)
    recent=compare_endpoint_pair(scope,fresh_views['reference'],fresh_views['source_visible_01'])
    assert recent['decision']=='reference_endpoint_pair_candidate' and recent['middle_connection']=='unobserved'
    assert source_pins()==p['mainline_pins']
    verify(pins)
    save(OUT/'report.json',dict(status='complete',cases=results,recent_fresh_SAM_inventory_replay=recent,
        fresh_model_calls_this_trial=0,original_masks_unchanged=True,mainline_pins_unchanged=True,
        actual_visual_review='pending',new_physical_wire_identities_confirmed=0,
        new_electrical_connections_confirmed=0,deployed=False,not_field_accuracy=True,
        expected_missing_middle_not_inferred=True))
    print(json.dumps([(r['id'],r['decision'],r['middle_connection'],r['automatic_topology_comparison']['decision']) for r in results]))


if __name__=='__main__':main()
