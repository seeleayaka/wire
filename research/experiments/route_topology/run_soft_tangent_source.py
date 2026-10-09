"""Five-source tolerant pairing contrast; old native mask acceptance unchanged."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from soft_tangent_candidates import PARAMS,observe_family
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins
from run_prompt_contrast import digest,save,verify

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/soft_tangent_source_20261008'


def main():
    old=ROOT/'artifacts/tangent_gap_native_bound_20261008'
    previous=json.loads((old/'report.json').read_text(encoding='utf-8'))
    p=json.loads((old/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'))
    poses={r['id']:r for r in prep['cases']}
    bp=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(bp['pins'])
    scope=json.loads(Path(bp['confirmed_scope_path']).read_text(encoding='utf-8'))
    before=source_pins();assert before==p['mainline_pins']
    files=[Path(__file__),Path(__file__).with_name('soft_tangent_candidates.py'),Path(__file__).with_name('tangent_gap_candidates.py'),
           old/'report.json',old/'protocol.json',ROOT/'artifacts/SOFT_TANGENT_SOURCE_PROTOCOL_20261008.md',
           base/'preparation_report.json',Path(__file__).with_name('run_reference_color_paths_source.py')]
    pins={str(f):digest(f) for f in files};pins.update(p['pins']);verify(pins)
    OUT.mkdir(exist_ok=False);save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,params=PARAMS,
         native_SAM_acceptance=.75,source_only=True,old_SAM_and_pose_reused=True,masks_not_repaired=True))
    rows=[]
    for oldrow in previous['cases']:
        source=oldrow['source_binding'];assert digest(source['path'])==source['image_sha256']
        crop=oldrow['crop_box_xyxy'];rgb=np.asarray(Image.open(source['path']).convert('RGB').crop(crop))
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10;colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
        regions=exact_regions(rgb.shape[:2],crop,scope,poses[oldrow['id']]['anchors']);assert regions is not None
        records=[]
        for oldrec in oldrow['records']:
            assert oldrec['score']>=.75 and digest(oldrec['mask_path'])==oldrec['mask_sha256']
            mask=np.asarray(Image.open(oldrec['mask_path']).convert('L'))>=128
            groups=[];canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
            for oldfamily in oldrec['families']:
                family=oldfamily['hue_bins'];raw=colored&mask&np.isin(bins,family)
                result,_=observe_family(raw,regions);result['hue_bins']=family;groups.append(result)
                Image.fromarray(raw.astype('uint8')*255).save(OUT/(oldrow['id']+'_'+oldrec['record_id']+'_family_'+str(family[0])+'.png'))
                for edge in result['edges']:
                    if edge['state']!='fragment_pair_candidate':continue
                    pts=edge['virtual_curve_xy']
                    for i in range(0,len(pts)-1,4):draw.line([tuple(pts[i]),tuple(pts[min(i+1,len(pts)-1)])],fill='magenta',width=2)
                for e in result['endpoints']:
                    x,y=e['point_xy'];u,v=e['outward_unit'];draw.line([(x,y),(x+u*12,y+v*12)],fill='cyan')
            ok=all(f['anchor_path_candidates'] for f in groups)
            records.append(dict(record_id=oldrec['record_id'],score=oldrec['score'],mask_path=oldrec['mask_path'],mask_sha256=oldrec['mask_sha256'],
                  families=groups,soft_two_family_path_candidate=ok,old_hard_path=oldrec['tangent_two_family_path_candidate']))
            w,h=canvas.size;display=Image.new('RGB',(max(w,600),h+55),'white');display.paste(canvas,(0,0));d=ImageDraw.Draw(display)
            d.text((8,h+8),oldrow['id']+' soft_path='+str(ok),fill='black')
            d.text((8,h+30),'cyan: multiscale tangent; dashed: inferred ONLY',fill='black');display.save(OUT/(oldrow['id']+'_'+oldrec['record_id']+'.png'))
        rows.append(dict(id=oldrow['id'],source_binding=source,crop_box_xyxy=crop,original_socket_state=oldrow['original_socket_state'],records=records,
             old_hard_path=oldrow['tangent_path'],soft_path=any(r['soft_two_family_path_candidate'] for r in records),
             old_native_path=oldrow['native_path'],observed_pixels_added=0,physical_identity_confirmed=False,electrical_continuity='not_assessed'))
    visible=[r for r in rows if r['original_socket_state']=='mating_body_visible']
    gains=[r['id'] for r in visible if not r['old_hard_path'] and r['soft_path']]
    losses=[r['id'] for r in visible if r['old_hard_path'] and not r['soft_path']]
    unsafe=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['soft_path']]
    gate=bool(gains) and not losses and not unsafe and all(r['soft_path'] for r in visible)
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=rows,source_gate_passed=gate,gains=gains,losses=losses,exposed_source_candidates=unsafe,
         fragment_candidates=sum(e['state']=='fragment_pair_candidate' for r in rows for rec in r['records'] for f in rec['families'] for e in f['edges']),
         total_geometric_pairs=sum(len(f['edges']) for r in rows for rec in r['records'] for f in rec['families']),
         mainline_unchanged=True,new_confirmed_connections=0,new_confirmed_wire_identities=0,
         fresh_SAM_calls=0,accuracy_not_measured=True,deployed=False,demo_images_read=False)
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    print(json.dumps([dict(id=r['id'],old=r['old_hard_path'],soft=r['soft_path']) for r in rows]))


if __name__=='__main__':main()
