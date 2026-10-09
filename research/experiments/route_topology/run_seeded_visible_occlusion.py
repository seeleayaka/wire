"""Source-only supplemental RGB evidence, distinct from pinned SAM footprint."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from seeded_visible_color import supplement
from soft_tangent_candidates import observe_family
from occluder_tangent_candidates import augment_edges
from run_occluder_tangent_source import paths
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins
from run_prompt_contrast import digest,save,verify

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/seeded_visible_occlusion_20261008'


def main():
    base=ROOT/'artifacts/soft_tangent_source_20261008';p=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    old=json.loads((base/'report.json').read_text(encoding='utf-8'));before=source_pins();assert before==p['mainline_pins']
    bp=ROOT/'artifacts/mendeley_cable_socket_controls_20261006';prep=json.loads((bp/'preparation_report.json').read_text(encoding='utf-8'));poses={c['id']:c for c in prep['cases']}
    scope=json.loads(Path(json.loads((bp/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']).read_text(encoding='utf-8'))
    files=[Path(__file__),Path(__file__).with_name('seeded_visible_color.py'),Path(__file__).with_name('occluder_tangent_candidates.py'),Path(__file__).with_name('run_occluder_tangent_source.py'),
        base/'report.json',base/'protocol.json',ROOT/'artifacts/SEEDED_VISIBLE_OCCLUSION_PROTOCOL_20261008.md',bp/'preparation_report.json']
    pins=dict(p['pins']);pins.update({str(f):digest(f) for f in files});verify(pins);OUT.mkdir(exist_ok=False)
    save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,seed_overlap_pixels=24,RGB_is_supplement_not_SAM=True,old_SAM_masks_and_pose_reused=True))
    rows=[]
    for row in old['cases']:
        assert digest(row['source_binding']['path'])==row['source_binding']['image_sha256']
        rgb=np.asarray(Image.open(row['source_binding']['path']).convert('RGB').crop(row['crop_box_xyxy']));regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors'])
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10;records=[]
        for rec in row['records']:
            assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
            native=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128
            selected,stats=supplement(rgb,native,sorted({b for f in rec['families'] for b in f['hue_bins']}))
            Image.fromarray(selected.astype('uint8')*255).save(OUT/(row['id']+'_'+rec['record_id']+'_RGB_supplement.png'))
            display=rgb.copy();extra=selected&~native;display[extra]=(display[extra]*.4+np.array([255,180,0])*.6).astype('uint8')
            canvas=Image.fromarray(display);draw=ImageDraw.Draw(canvas);families=[]
            for oldfamily in rec['families']:
                raw=selected&np.isin(bins,oldfamily['hue_bins']);result,_=observe_family(raw,regions)
                for e in result['edges']:e['state']='unaccepted_RGB_edge'
                edges=augment_edges(result['endpoints'],result['edges'],rgb,raw);newpaths=paths(raw,edges,regions)
                families.append(dict(hue_bins=oldfamily['hue_bins'],endpoints=result['endpoints'],edges=edges,supplemental_paths=newpaths,
                    old_SAM_family_paths=oldfamily['anchor_path_candidates'],combined_family_candidate=bool(oldfamily['anchor_path_candidates'] or newpaths),
                    RGB_anchor_hits={k:int((raw&r).sum()) for k,r in regions.items()}))
                for e in edges:
                    if e['state']!='occluder_supported_fragment_candidate':continue
                    curve=e['virtual_curve_xy']
                    for i in range(0,len(curve)-1,4):draw.line([tuple(curve[i]),tuple(curve[min(i+1,len(curve)-1)])],fill='lime',width=2)
            records.append(dict(record_id=rec['record_id'],mask_path=rec['mask_path'],mask_sha256=rec['mask_sha256'],score=rec['score'],supplement=stats,
                families=families,new_path=all(f['combined_family_candidate'] for f in families)))
            w,h=canvas.size;output=Image.new('RGB',(max(w,630),h+55),'white');output.paste(canvas,(0,0));d=ImageDraw.Draw(output)
            d.text((8,h+8),row['id']+' supplementary_path='+str(records[-1]['new_path']),fill='black')
            d.text((8,h+30),'orange: RGB not SAM; green dashed: inferred; NO wire identity',fill='black');output.save(OUT/(row['id']+'_'+rec['record_id']+'.png'))
        rows.append(dict(id=row['id'],source_binding=row['source_binding'],crop_box_xyxy=row['crop_box_xyxy'],original_socket_state=row['original_socket_state'],
            records=records,old_path=row['soft_path'],new_path=any(r['new_path'] for r in records)))
    gains=[r['id'] for r in rows if not r['old_path'] and r['new_path']];losses=[r['id'] for r in rows if r['old_path'] and not r['new_path']]
    exposed=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['new_path']]
    gate=bool(gains) and not losses and not exposed and all(r['new_path'] for r in rows if r['original_socket_state']=='mating_body_visible')
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_source_candidates=exposed,source_gate_passed=gate,
        added_supplementary_fragment_candidates=sum(e['state']=='occluder_supported_fragment_candidate' for r in rows for rec in r['records'] for f in rec['families'] for e in f['edges']),
        original_SAM_pixels_modified=0,recovered_RGB_pixels=sum(rec['supplement']['recovered_RGB_pixels'] for r in rows for rec in r['records']),
        new_confirmed_connections=0,physical_identity_confirmed=False,electrical_continuity='not_assessed',fresh_SAM_calls=0,deployed=False,mainline_unchanged=True)
    save(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    print(json.dumps([dict(id=r['id'],old=r['old_path'],new=r['new_path'],hits=[f['RGB_anchor_hits'] for rec in r['records'] for f in rec['families']]) for r in rows]))


if __name__=='__main__':main()
