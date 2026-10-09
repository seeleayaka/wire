"""Source-only native RGB path feasibility, never changes SAM acceptance."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from reference_color_paths import hue_families,observe_paths
from run_prompt_contrast import save,digest,verify
from run_review import verified_run
from visible_bundle_relation import observe_bundle
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/reference_color_paths_source_20261008'


def exact_regions(shape,crop,scope,poses):
    ys,xs=np.indices(shape);pts=np.column_stack((xs.ravel()+crop[0],ys.ravel()+crop[1],np.ones(xs.size)))
    regions={}
    for anchor in scope['anchors']:
        pose=next(p for p in poses if p['id']==anchor['id'])
        if not pose['localization_proposal_supported'] or not all(pose['gates'].values()):return None
        q=pts@np.asarray(pose['inspection_to_reference_local']).T
        if (abs(q[:,2])<1e-9).any():raise ValueError('mapping horizon')
        xy=q[:,:2]/q[:,2:];l,t,r,b=anchor['bbox_xyxy']
        regions[anchor['id']]=((xy[:,0]>=l)&(xy[:,0]<=r)&(xy[:,1]>=t)&(xy[:,1]<=b)).reshape(shape)
    return regions


def main():
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prepared=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'))
    p=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    appearance=ROOT/'artifacts/wire_appearance_reference_v3_20261008/protocol.json'
    app=json.loads(appearance.read_text(encoding='utf-8'));verify(app['pins'])
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'))
    allowed=sorted({b for profile in app['reference_profiles'].values() for b in profile['allowed_hue_bins']})
    families=hue_families(allowed)
    before=source_pins();assert before==p['mainline_pins']
    files=[Path(__file__),Path(__file__).with_name('reference_color_paths.py'),base/'protocol.json',
           base/'preparation_report.json',appearance,ROOT/'artifacts/COLOR_PATH_SOURCE_PROTOCOL_20261008.md']
    pins={str(f):digest(f) for f in files};OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,hue_families=families,
        reference_only_palette=True,source_case_count=5,demo_images_read=False,
        fresh_SAM_calls=0,no_pixel_repair=True,not_a_physical_identity_model=True))
    cases=[]
    for row in prepared['cases']:
        source=row['original_source'];assert digest(source['path'])==source['image_sha256']
        crop=row['crop_context']['crop_box_xyxy'];rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(crop))
        assert np.array_equal(rgb,np.asarray(Image.open(row['crop_context']['source']['path']).convert('RGB')))
        regions=exact_regions(rgb.shape[:2],crop,scope,row['anchors'])
        if regions is None:raise ValueError('source prerequisites unavailable')
        result,pixels=observe_paths(rgb,regions,families)
        inv=verified_run(base/row['id']/'cable_plus_reference_anatomy_box',row['crop_context']['source']['path'])
        masks=[dict(raw=np.asarray(Image.open(path).convert('L')),score=float(score),record_id=path.stem,
                    recipe='cable_plus_reference_anatomy_box') for path,score in zip(inv['paths'],inv['scores'])]
        baseline=observe_bundle(scope,{k:source[k] for k in ['image_sha256','image_size','coordinate_frame']},
            row['anchors'],masks,row['phenotype'],translation=tuple(crop[:2]),sam_inventory_verified=True)
        cases.append(dict(id=row['id'],source_binding=source,old_native_visible_component=baseline['unique_native_bundle_observation_supported'],
            source_socket_state=row['phenotype'],color_paths=result))
        display=rgb.copy();display[pixels]=(display[pixels]*.4+np.array([0,200,220])*.6).astype('uint8')
        h,w=rgb.shape[:2];canvas=Image.new('RGB',(max(w,530),h+65),'white');canvas.paste(Image.fromarray(display),(0,0))
        draw=ImageDraw.Draw(canvas);draw.text((8,h+8),row['id']+' RGB paths only; NO electrical identity',fill='black')
        draw.text((8,h+30),result['state']+'; no gaps closed',fill='black');canvas.save(OUT/(row['id']+'.png'))
        if row['id']=='reference' and result['state']!='native_multicolor_path_candidate':break
    visible=[r for r in cases if r['source_socket_state']=='mating_body_visible' and r['id']!='reference']
    gains=[r['id'] for r in visible if not r['old_native_visible_component'] and r['color_paths']['state']=='native_multicolor_path_candidate']
    losses=[r['id'] for r in visible if r['old_native_visible_component'] and r['color_paths']['state']!='native_multicolor_path_candidate']
    unsafe=[r['id'] for r in cases if r['source_socket_state']=='socket_contacts_exposed' and r['color_paths']['state']=='native_multicolor_path_candidate']
    passed=len(cases)==5 and cases[0]['color_paths']['state']=='native_multicolor_path_candidate' and bool(gains) and not losses and not unsafe
    report=dict(status='complete',source_cases_evaluated=len(cases),planned_source_cases=5,cases=cases,
        strict_source_gain=passed,gains=gains,old_visible_support_losses=losses,exposed_color_path_candidates=unsafe,
        next_action='independent source audit before any demo' if passed else 'reject; no demo evaluation',
        new_confirmed_wire_identities=0,new_electrical_connections=0,demo_inference_run=False,not_deployed=True,fresh_SAM_calls=0)
    verify(pins);assert source_pins()==before;save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
