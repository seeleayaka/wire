"""Original five sources; frozen local tangent association, not fault verdict."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from tangent_gap_candidates import PARAMS,observe_family
from reference_color_paths import hue_families,observe_paths
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins
from run_prompt_contrast import digest,save,verify

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/tangent_gap_source_20261008'


def main():
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'))
    old_protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(old_protocol['pins'])
    scope=json.loads(Path(old_protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    app_path=ROOT/'artifacts/wire_appearance_reference_v3_20261008/protocol.json'
    appearance=json.loads(app_path.read_text(encoding='utf-8'));verify(appearance['pins'])
    families=hue_families(sorted({b for p in appearance['reference_profiles'].values() for b in p['allowed_hue_bins']}))
    before=source_pins();assert before==old_protocol['mainline_pins']
    files=[Path(__file__),Path(__file__).with_name('tangent_gap_candidates.py'),
           Path(__file__).with_name('reference_color_paths.py'),Path(__file__).with_name('run_reference_color_paths_source.py'),
           base/'protocol.json',base/'preparation_report.json',app_path,
           ROOT/'artifacts/TANGENT_GAP_SOURCE_PROTOCOL_20261008.md']
    pins={str(f):digest(f) for f in files}
    for row in prep['cases']:pins[row['original_source']['path']]=row['original_source']['image_sha256']
    verify(pins);OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,params=PARAMS,hue_families=families,
         source_only=True,demo_images_read=False,old_pose_reused=True,fresh_SAM_calls=0,
         same_RGB_observer=True,pairing_defaults_not_calibrated=True))
    cases=[]
    for row in prep['cases']:
        crop=row['crop_context']['crop_box_xyxy'];source=row['original_source']
        rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(crop))
        regions=exact_regions(rgb.shape[:2],crop,scope,row['anchors'])
        if regions is None:raise ValueError('unqualified source pose')
        native,_=observe_paths(rgb,regions,families)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)*18//180
        colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
        results=[];canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
        for family in families:
            raw=colored&np.isin(bins,family);original=raw.copy()
            result,skel=observe_family(raw,regions);assert np.array_equal(raw,original)
            result['hue_bins']=family;results.append(result)
            Image.fromarray(raw.astype('uint8')*255).save(OUT/(row['id']+'_family_'+str(family[0])+'.png'))
            for endpoint in result['endpoints']:
                x,y=endpoint['point_xy'];u,v=endpoint['outward_unit']
                draw.ellipse([x-2,y-2,x+2,y+2],outline='cyan')
                draw.line([(x,y),(x+u*10,y+v*10)],fill='cyan',width=1)
            for edge in result['edges']:
                points=edge['virtual_curve_xy'];color='magenta' if edge['state']=='fragment_pair_candidate' else 'orange'
                # Deliberately dashed: these pixels are inferred, never observed.
                for i in range(0,len(points)-1,4):draw.line([tuple(points[i]),tuple(points[min(i+1,len(points)-1)])],fill=color,width=2)
        paths=sum(bool(f['anchor_path_candidates']) for f in results)
        state='inferred_multicolor_path_candidate' if paths>=2 else 'insufficient_fragment_association'
        cases.append(dict(id=row['id'],source_binding=source,crop_box_xyxy=crop,
             original_socket_state=row['phenotype'],native_color_state=native['state'],
             tangent_state=state,families=results,physical_identity_confirmed=False,
             electrical_continuity='not_assessed',old_decisions_changed=False,observed_pixels_added=0))
        w,h=canvas.size;display=Image.new('RGB',(max(w,640),h+75),'white');display.paste(canvas,(0,0));d=ImageDraw.Draw(display)
        d.text((8,h+8),row['id']+' '+state,fill='black')
        d.text((8,h+30),'cyan: local tangent; dashed magenta: candidate; orange: ambiguous',fill='black')
        d.text((8,h+50),'INFERRED ONLY - no electrical connection confirmed',fill='black');display.save(OUT/(row['id']+'_proposals.png'))
        save(OUT/'progress.json',dict(status='running',completed=len(cases),total=len(prep['cases'])))
    visible=[c for c in cases if c['original_socket_state']=='mating_body_visible']
    gains=[c['id'] for c in visible if c['native_color_state']!='native_multicolor_path_candidate' and c['tangent_state']=='inferred_multicolor_path_candidate']
    unsafe=[c['id'] for c in cases if c['original_socket_state']=='socket_contacts_exposed' and c['tangent_state']=='inferred_multicolor_path_candidate']
    passed=bool(gains) and all(c['tangent_state']=='inferred_multicolor_path_candidate' for c in visible) and not unsafe
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=cases,source_gate_passed=passed,
         tentative_source_gain_ids=gains,exposed_source_path_candidate_ids=unsafe,
         source_cases_evaluated=5,source_families_evaluated=10,geometry_pair_candidates=sum(
           e['state']=='fragment_pair_candidate' for c in cases for f in c['families'] for e in f['edges']),
         ambiguous_or_nonreciprocal_pairs=sum(e['state']!='fragment_pair_candidate' for c in cases for f in c['families'] for e in f['edges']),
         new_confirmed_wire_identities=0,new_electrical_connections=0,accuracy_not_measured=True,
         virtual_edges_not_mask_pixels=True,mainline_unchanged=True,deployed=False,
         fresh_SAM_calls=0,seconds_not_training=True,demo_images_read=False,
         next_action='independent source audit before expansion' if passed else 'reject expansion; retain local pairing diagnostics')
    save(OUT/'report.json',report);save(OUT/'progress.json',dict(status='complete',completed=5,total=5))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    print(json.dumps([dict(id=c['id'],state=c['tangent_state'],families=[dict(hue_bins=f['hue_bins'],endpoints=len(f['endpoints']),pairs=len(f['edges']),paths=len(f['anchor_path_candidates'])) for f in c['families']]) for c in cases]))


if __name__=='__main__':main()
