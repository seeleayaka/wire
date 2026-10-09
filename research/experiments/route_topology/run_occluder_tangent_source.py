"""Five original full-context sources, witnessed virtual proposals only."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from occluder_tangent_candidates import augment_edges,OBSTACLE_PARAMS
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins
from run_prompt_contrast import digest,save,verify

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/occluder_tangent_source_20261008'


def paths(raw,edges,regions):
    n,labels=cv2.connectedComponents(raw.astype('uint8'),connectivity=8);adj={i:set() for i in range(1,n)}
    for edge in edges:
        if edge['state'] in ['fragment_pair_candidate','occluder_supported_fragment_candidate']:
            a,b=edge['components'];adj[a].add(b);adj[b].add(a)
    found=[];seen=set()
    for start in adj:
        if start in seen:continue
        todo=[start];group=set()
        while todo:
            c=todo.pop()
            if c in group:continue
            group.add(c);todo.extend(adj[c]-group)
        seen|=group;selection=np.isin(labels,list(group));hits={k:int((selection&r).sum()) for k,r in regions.items()}
        if all(v>=8 for v in hits.values()):found.append(dict(components=sorted(group),anchor_hits=hits,contains_inferred_gap=len(group)>1,physical_identity_confirmed=False))
    return found


def main():
    base=ROOT/'artifacts/soft_tangent_source_20261008';p=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    old=json.loads((base/'report.json').read_text(encoding='utf-8'));before=source_pins();assert before==p['mainline_pins']
    prepbase=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    prep=json.loads((prepbase/'preparation_report.json').read_text(encoding='utf-8'));poses={c['id']:c for c in prep['cases']}
    scope=json.loads(Path(json.loads((prepbase/'protocol.json').read_text(encoding='utf-8'))['confirmed_scope_path']).read_text(encoding='utf-8'))
    files=[Path(__file__),Path(__file__).with_name('occluder_tangent_candidates.py'),base/'report.json',base/'protocol.json',ROOT/'artifacts/OCCLUDER_TANGENT_PROTOCOL_20261008.md',prepbase/'preparation_report.json']
    pins=dict(p['pins']);pins.update({str(f):digest(f) for f in files});verify(pins)
    OUT.mkdir(exist_ok=False);save(OUT/'protocol.json',dict(pins=pins,mainline_pins=before,obstacle_params=OBSTACLE_PARAMS,same_model_observer_count=1,new_SAM_calls=0))
    rows=[];partial=[]
    for row in old['cases']:
        assert digest(row['source_binding']['path'])==row['source_binding']['image_sha256']
        rgb=np.asarray(Image.open(row['source_binding']['path']).convert('RGB').crop(row['crop_box_xyxy']))
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);bins=hsv[:,:,0].astype(int)//10;colored=(hsv[:,:,1]>=64)&(hsv[:,:,2]>=32)
        regions=exact_regions(rgb.shape[:2],row['crop_box_xyxy'],scope,poses[row['id']]['anchors']);records=[]
        for rec in row['records']:
            assert rec['score']>=.75 and digest(rec['mask_path'])==rec['mask_sha256']
            mask=np.asarray(Image.open(rec['mask_path']).convert('L'))>=128;canvas=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(canvas);families=[]
            for family in rec['families']:
                raw=colored&mask&np.isin(bins,family['hue_bins']);edges=augment_edges(family['endpoints'],family['edges'],rgb,raw)
                newpaths=paths(raw,edges,regions);oldpath=bool(family['anchor_path_candidates']);newpath=bool(newpaths)
                assert not oldpath or newpath,'old graph support lost'
                if newpath and not oldpath:partial.append(dict(id=row['id'],record_id=rec['record_id'],hue_bins=family['hue_bins']))
                added=[e for e in edges if e['state']=='occluder_supported_fragment_candidate']
                families.append(dict(hue_bins=family['hue_bins'],endpoints=family['endpoints'],edges=edges,anchor_path_candidates=newpaths,
                    old_family_path=oldpath,added_fragment_candidates=len(added),raw_anchor_hits={k:int((raw&r).sum()) for k,r in regions.items()}))
                for e in edges:
                    if e['state'] not in ['fragment_pair_candidate','occluder_supported_fragment_candidate']:continue
                    color='lime' if e['state']=='occluder_supported_fragment_candidate' else 'magenta';curve=e['virtual_curve_xy']
                    for i in range(0,len(curve)-1,4):draw.line([tuple(curve[i]),tuple(curve[min(i+1,len(curve)-1)])],fill=color,width=2)
            records.append(dict(record_id=rec['record_id'],score=rec['score'],mask_path=rec['mask_path'],mask_sha256=rec['mask_sha256'],families=families,new_path=all(f['anchor_path_candidates'] for f in families)))
            w,h=canvas.size;display=Image.new('RGB',(max(w,620),h+55),'white');display.paste(canvas,(0,0));d=ImageDraw.Draw(display)
            d.text((8,h+8),row['id']+' path_candidate='+str(records[-1]['new_path']),fill='black');d.text((8,h+30),'dashed magenta: prior; green: new inferred; NO electrical verdict',fill='black')
            display.save(OUT/(row['id']+'_'+rec['record_id']+'.png'))
        rows.append(dict(id=row['id'],source_binding=row['source_binding'],crop_box_xyxy=row['crop_box_xyxy'],original_socket_state=row['original_socket_state'],
            records=records,old_path=row['soft_path'],new_path=any(r['new_path'] for r in records)))
    gains=[r['id'] for r in rows if not r['old_path'] and r['new_path']];losses=[r['id'] for r in rows if r['old_path'] and not r['new_path']]
    exposed=[r['id'] for r in rows if r['original_socket_state']=='socket_contacts_exposed' and r['new_path']]
    gate=bool(gains) and not losses and not exposed and all(r['new_path'] for r in rows if r['original_socket_state']=='mating_body_visible')
    verify(pins);assert source_pins()==before
    report=dict(status='complete',cases=rows,gains=gains,losses=losses,exposed_source_candidates=exposed,source_gate_passed=gate,partial_family_path_gains=partial,
        added_fragment_candidates=sum(f['added_fragment_candidates'] for r in rows for rec in r['records'] for f in rec['families']),
        observed_pixels_added=0,new_confirmed_connections=0,physical_identity_confirmed=False,electrical_continuity='not_assessed',mainline_unchanged=True,deployed=False)
    save(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
