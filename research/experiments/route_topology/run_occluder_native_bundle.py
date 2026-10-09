"""Whole native SAM bundle fragments, no color-induced breaks or pixel repair."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from soft_tangent_candidates import observe_family
from occluder_tangent_candidates import augment_edges,OBSTACLE_PARAMS
from run_occluder_tangent_source import paths
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins
from run_prompt_contrast import digest,save,verify

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/occluder_native_bundle_20261008'


def main():
    base=ROOT/'artifacts/soft_tangent_source_20261008';p=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    old=json.loads((base/'report.json').read_text(encoding='utf-8'));before=source_pins();assert before==p['mainline_pins']
    bp=ROOT/'artifacts/mendeley_cable_socket_controls_20261006';prep=json.loads((bp/'preparation_report.json').read_text(encoding='utf-8'));poses={c['id']:c for c in prep['cases']}
    scope=json.loads(Path(json.loads((bp/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']).read_text(encoding='utf-8'))
    files=[Path(__file__),Path(__file__).with_name('occluder_tangent_candidates.py'),Path(__file__).with_name('run_occluder_tangent_source.py'),
        base/'report.json',base/'protocol.json',ROOT/'artifacts/OCCLUDER_NATIVE_BUNDLE_PROTOCOL_20261008.md',bp/'preparation_report.json']
    pins=dict(p['pins']);pins.update({str(f):digest(f) for f in files});verify(pins);OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,obstacle_params=OBSTACLE_PARAMS,native_SAM_acceptance=.75,
        representation='single_native_SAM_bundle_not_single_wire',pose_and_masks_reused=True))
    rows=[]
    for row in old['cases']:
        assert digest(row['source_binding']['path'])==row['source_binding']['image_sha256']
        rgb=np.asarray(Image.open(row['source_binding']['path']).convert('RGB').crop(row['crop_box_xyxy']));regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors']);records=[]
        for rec in row['records']:
            assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
            raw=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128;result,_=observe_family(raw,regions)
            oldpaths=paths(raw,[],regions)
            for e in result['edges']:e['state']='unaccepted_native_bundle_edge'
            edges=augment_edges(result['endpoints'],result['edges'],rgb,raw);newpaths=paths(raw,edges,regions)
            assert not oldpaths or newpaths
            records.append(dict(record_id=rec['record_id'],score=rec['score'],mask_path=rec['mask_path'],mask_sha256=rec['mask_sha256'],
                endpoints=result['endpoints'],edges=edges,native_components=result['native_components'],old_native_paths=oldpaths,new_paths=newpaths,
                raw_anchor_hits={k:int((raw&r).sum()) for k,r in regions.items()},observed_pixels_added=0))
            canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
            for e in edges:
                if e['state']!='occluder_supported_fragment_candidate':continue
                curve=e['virtual_curve_xy']
                for i in range(0,len(curve)-1,4):draw.line([tuple(curve[i]),tuple(curve[min(i+1,len(curve)-1)])],fill='lime',width=2)
            for end in result['endpoints']:
                x,y=end['point_xy'];draw.ellipse((x-1,y-1,x+1,y+1),fill='cyan')
            for k,r in regions.items():
                yy,xx=np.nonzero(r);draw.rectangle((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())),outline='orange')
            w,h=canvas.size;display=Image.new('RGB',(max(w,620),h+55),'white');display.paste(canvas,(0,0));d=ImageDraw.Draw(display)
            d.text((8,h+8),row['id']+' native='+str(bool(oldpaths))+' inferred_bundle='+str(bool(newpaths)),fill='black')
            d.text((8,h+30),'green dashed: inferred ONLY; cyan: native ends; NO single-wire identity',fill='black');display.save(OUT/(row['id']+'_'+rec['record_id']+'.png'))
        rows.append(dict(id=row['id'],source_binding=row['source_binding'],crop_box_xyxy=row['crop_box_xyxy'],original_socket_state=row['original_socket_state'],
            records=records,old_path=any(r['old_native_paths'] for r in records),new_path=any(r['new_paths'] for r in records)))
    gains=[r['id'] for r in rows if not r['old_path'] and r['new_path']];losses=[r['id'] for r in rows if r['old_path'] and not r['new_path']]
    exposed=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['new_path']]
    gate=bool(gains) and not losses and not exposed and all(r['new_path'] for r in rows if r['original_socket_state']=='mating_body_visible')
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_source_candidates=exposed,source_gate_passed=gate,
        added_fragment_candidates=sum(e['state']=='occluder_supported_fragment_candidate' for r in rows for rec in r['records'] for e in rec['edges']),
        observed_pixels_added=0,new_confirmed_connections=0,physical_identity_confirmed=False,electrical_continuity='not_assessed',fresh_SAM_calls=0,deployed=False,mainline_unchanged=True)
    save(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    print(json.dumps([dict(id=r['id'],old=r['old_path'],new=r['new_path']) for r in rows]))


if __name__=='__main__':main()
