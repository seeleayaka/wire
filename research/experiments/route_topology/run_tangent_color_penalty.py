"""Single color-penalty contrast on five original sources, no new SAM call."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tangent_color_penalty import observe_family
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_reference_color_paths_source import exact_regions

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/tangent_color_penalty_20261008'


def main():
    base=ROOT/'artifacts/soft_tangent_source_20261008';p=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    old=json.loads((base/'report.json').read_text(encoding='utf-8'))
    source=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((source/'preparation_report.json').read_text(encoding='utf-8'));poses={r['id']:r for r in prep['cases']}
    scope_path=json.loads((source/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']
    scope=json.loads(Path(scope_path).read_text(encoding='utf-8'));before=source_pins();assert before==p['mainline_pins']
    files=[Path(__file__),Path(__file__).with_name('tangent_color_penalty.py'),base/'report.json',base/'protocol.json',ROOT/'artifacts/TANGENT_COLOR_PENALTY_PROTOCOL_20261008.md']
    pins={str(f):digest(f) for f in files};pins.update(p['pins']);verify(pins)
    OUT.mkdir(exist_ok=False);save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,color_penalty=.25,geometry_params=p['params'],masks_not_repaired=True))
    rows=[]
    for row in old['cases']:
        binding=row['source_binding'];assert digest(binding['path'])==binding['image_sha256']
        rgb=np.asarray(Image.open(binding['path']).convert('RGB').crop(row['crop_box_xyxy']))
        regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors']);records=[]
        for rec in row['records']:
            groups=[];canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas)
            for f in rec['families']:
                rawpath=base/(row['id']+'_'+rec['record_id']+'_family_'+str(f['hue_bins'][0])+'.png')
                raw=np.asarray(Image.open(rawpath).convert('L'))>0
                result,_=observe_family(raw,regions,rgb);result['hue_bins']=f['hue_bins'];result['old_family_path']=bool(f['anchor_path_candidates']);groups.append(result)
                assert result['native_components']==f['native_components']
                for e in result['edges']:
                    if e['state']!='fragment_pair_candidate':continue
                    pts=e['virtual_curve_xy']
                    for i in range(0,len(pts)-1,4):draw.line([tuple(pts[i]),tuple(pts[min(i+1,len(pts)-1)])],fill='magenta',width=2)
            records.append(dict(record_id=rec['record_id'],families=groups,old_path=rec['soft_two_family_path_candidate'],new_path=all(f['anchor_path_candidates'] for f in groups)))
            w,h=canvas.size;display=Image.new('RGB',(max(w,600),h+55),'white');display.paste(canvas,(0,0));d=ImageDraw.Draw(display)
            d.text((8,h+8),row['id']+' color_path='+str(records[-1]['new_path']),fill='black');d.text((8,h+30),'non-negative color penalty; dashed INFERRED only',fill='black');display.save(OUT/(row['id']+'_'+rec['record_id']+'.png'))
        rows.append(dict(id=row['id'],source_binding=binding,records=records,original_socket_state=row['original_socket_state'],old_path=row['soft_path'],new_path=any(r['new_path'] for r in records)))
    gains=[r['id'] for r in rows if not r['old_path'] and r['new_path']];losses=[r['id'] for r in rows if r['old_path'] and not r['new_path']]
    visible=[r for r in rows if r['original_socket_state']=='mating_body_visible'];unsafe=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['new_path']]
    gate=bool(gains) and not losses and not unsafe and all(r['new_path'] for r in visible)
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_source_path_candidates=unsafe,source_gate_passed=gate,
       candidates=sum(e['state']=='fragment_pair_candidate' for r in rows for rec in r['records'] for f in rec['families'] for e in f['edges']),
       new_confirmed_connections=0,physical_identity_confirmed=False,electrical_continuity='not_assessed',
       fresh_SAM_calls=0,deployed=False,demo_images_read=False,mainline_unchanged=True)
    save(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    print(json.dumps([dict(id=r['id'],old=r['old_path'],new=r['new_path'],families=[dict(bins=f['hue_bins'],old=f['old_family_path'],new=bool(f['anchor_path_candidates']),rawhits=f['raw_anchor_hits']) for rec in r['records'] for f in rec['families']]) for r in rows]))


if __name__=='__main__':main()
