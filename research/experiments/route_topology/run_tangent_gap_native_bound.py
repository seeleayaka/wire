"""Single-variable contrast: geometry restricted to each high-score native SAM."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tangent_gap_candidates import PARAMS,observe_family
from reference_color_paths import hue_families
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins
from run_review import verified_run
from run_prompt_contrast import digest,save,verify

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/tangent_gap_native_bound_20261008'


def main():
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'))
    prior=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(prior['pins'])
    first=json.loads((ROOT/'artifacts/tangent_gap_source_20261008/protocol.json').read_text(encoding='utf-8'));verify(first['pins'])
    before=source_pins();assert before==prior['mainline_pins']
    scope=json.loads(Path(prior['confirmed_scope_path']).read_text(encoding='utf-8'))
    families=first['hue_families']
    files=[Path(__file__),Path(__file__).with_name('tangent_gap_candidates.py'),
           Path(__file__).with_name('run_reference_color_paths_source.py'),base/'protocol.json',base/'preparation_report.json',
           ROOT/'artifacts/tangent_gap_source_20261008/protocol.json',ROOT/'artifacts/TANGENT_GAP_NATIVE_BOUND_PROTOCOL_20261008.md']
    pins={str(f):digest(f) for f in files};rows=[];inventories={}
    for row in prep['cases']:
        inv=verified_run(base/row['id']/'cable_plus_reference_anatomy_box',row['crop_context']['source']['path'])
        inventories[row['id']]=inv;pins.update({str(p):digest(p) for p in inv['paths']})
        manifest=Path(inv['directory'])/'run_manifest.json';pins[str(manifest)]=digest(manifest)
        pins[row['original_source']['path']]=row['original_source']['image_sha256']
    verify(pins);OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,params=PARAMS,hue_families=families,
         native_SAM_acceptance=.75,source_only=True,no_cross_instance_pairing=True,
         old_SAM_reused=True,fresh_SAM_calls=0,virtual_edges_not_observed=True))
    for row in prep['cases']:
        crop=row['crop_context']['crop_box_xyxy'];source=row['original_source']
        rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(crop))
        regions=exact_regions(rgb.shape[:2],crop,scope,row['anchors']);assert regions is not None
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)*18//180
        colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32);inventory=inventories[row['id']];records=[]
        for path,score in zip(inventory['paths'],inventory['scores']):
            if score<.75:continue
            mask=np.asarray(Image.open(path).convert('L'))>=128;assert mask.shape==rgb.shape[:2]
            groups=[];canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
            for family in families:
                raw=mask&colored&np.isin(bins,family);result,_=observe_family(raw,regions)
                result['hue_bins']=family;groups.append(result)
                Image.fromarray(raw.astype('uint8')*255).save(OUT/(row['id']+'_'+path.stem+'_family_'+str(family[0])+'.png'))
                for edge in result['edges']:
                    points=edge['virtual_curve_xy'];color='magenta' if edge['state']=='fragment_pair_candidate' else 'orange'
                    for i in range(0,len(points)-1,4):draw.line([tuple(points[i]),tuple(points[min(i+1,len(points)-1)])],fill=color,width=2)
                for endpoint in result['endpoints']:
                    x,y=endpoint['point_xy'];u,v=endpoint['outward_unit']
                    draw.ellipse([x-2,y-2,x+2,y+2],outline='cyan');draw.line([(x,y),(x+u*10,y+v*10)],fill='cyan')
            path_ok=all(g['anchor_path_candidates'] for g in groups)
            native_ok=all(any(not p['contains_inferred_gap'] for p in g['anchor_path_candidates']) for g in groups)
            records.append(dict(record_id=path.stem,score=score,mask_sha256=digest(path),mask_path=str(path),families=groups,
                  tangent_two_family_path_candidate=path_ok,native_two_family_path_candidate=native_ok))
            w,h=canvas.size;display=Image.new('RGB',(max(w,600),h+55),'white');display.paste(canvas,(0,0));d=ImageDraw.Draw(display)
            d.text((8,h+8),row['id']+' '+path.stem+' path_candidate='+str(path_ok),fill='black')
            d.text((8,h+30),'same SAM only; dashed INFERRED; no electrical identity',fill='black');display.save(OUT/(row['id']+'_'+path.stem+'.png'))
        rows.append(dict(id=row['id'],source_binding=source,crop_box_xyxy=crop,original_socket_state=row['phenotype'],records=records,
              native_path=any(r['native_two_family_path_candidate'] for r in records),
              tangent_path=any(r['tangent_two_family_path_candidate'] for r in records),
              observed_pixels_added=0,physical_identity_confirmed=False,electrical_continuity='not_assessed'))
    visible=[r for r in rows if r['original_socket_state']=='mating_body_visible']
    gains=[r['id'] for r in visible if not r['native_path'] and r['tangent_path']]
    losses=[r['id'] for r in visible if r['native_path'] and not r['tangent_path']]
    unsafe=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['tangent_path']]
    passed=bool(gains) and not losses and not unsafe and all(r['tangent_path'] for r in visible)
    verify(pins);verify(first['pins']);assert source_pins()==before
    report=dict(status='complete',cases=rows,source_gate_passed=passed,gains=gains,losses=losses,
         exposed_source_candidates=unsafe,high_score_instances=sum(len(r['records']) for r in rows),
         geometry_candidates=sum(e['state']=='fragment_pair_candidate' for r in rows for rec in r['records'] for f in rec['families'] for e in f['edges']),
         new_confirmed_wire_identities=0,new_electrical_connections=0,deployed=False,fresh_SAM_calls=0,
         demo_images_read=False,mainline_unchanged=True,no_accuracy_denominator=True,
         next_action='independent source audit before expansion' if passed else 'reject expansion; no tuning or deployment')
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    print(json.dumps([dict(id=r['id'],native=r['native_path'],tangent=r['tangent_path'],records=len(r['records'])) for r in rows]))


if __name__=='__main__':main()
