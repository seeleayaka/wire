import hashlib
from collections import Counter
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from inspection_agent.terminal_mapping import validate_mapping
from visible_entry_path import inspect_local_path
from directional_entry_path import inspect_directional_path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--directional',action='store_true');args=parser.parse_args()
OUT=ROOT/('artifacts/directional_entry_path_20261001' if args.directional else 'artifacts/visible_entry_path_20261001');OUT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
protocol=load(ROOT/'artifacts/sam_crop_coverage_20261001/frozen_protocol.json')
summaries=[]
for case in protocol['cases']:
    name=case['case'];run=Path(case['fresh_run_directory'])
    map_path=ROOT/f'artifacts/source_entry_drafts_20261001/{name}_entry_draft.json'
    map_digest=hashlib.sha256(map_path.read_bytes()).hexdigest();mapping=load(map_path)
    binding=validate_mapping(mapping,Path(mapping['image_binding']['image_path']))
    with Image.open(binding['image_path']) as image:rgb=np.array(image.convert('RGB'))
    report=load(ROOT/f'artifacts/sam_crop_coverage_20261001/{name}_source_endpoints.json')
    assert all(report['image_binding'][k]==binding[k] for k in ('image_sha256','image_size','coordinate_frame'))
    masks={};labels={};x1,y1,x2,y2=case['crop_box_xyxy']
    for row in report['records']:
        if not row['geometry_pair_eligible']:continue
        mid=row['source_mask_id']
        if mid not in labels:
            with Image.open(run/'sam'/f'mask_{mid:03d}.png') as image:binary=(np.array(image.convert('L'))>0).astype(np.uint8)
            labels[mid]=cv2.connectedComponents(binary,connectivity=8)[1]
        mask=np.zeros(rgb.shape[:2],dtype=bool)
        mask[y1:y2,x1:x2]=labels[mid]==row['component_id'];masks[row['record_id']]=mask
    audits=[]
    for port in mapping['ports']:
        candidates=[]
        for rid,mask in masks.items():
            result=(inspect_directional_path(rgb,mask,port['bbox_xyxy'],[0,1]) if args.directional
                    else inspect_local_path(rgb,mask,port['bbox_xyxy']))
            if result['continuous_pixel_support']:candidates.append(dict(record_id=rid,**result))
        # Every connected candidate retained. No winner selection by distance.
        audit=dict(port_id=port['id'],confirmed=False,candidates=candidates,
                   state='multiple_pixel_paths_ambiguous' if len(candidates)>1 else 'single_pixel_path_unconfirmed' if candidates else 'no_continuous_pixel_support_not_missing_wire')
        audits.append(audit)
        for index,candidate in enumerate(candidates):
            left,top,right,bottom=candidate['local_box_xyxy'];base=rgb[top:bottom,left:right]
            overlay=base.copy();active=masks[candidate['record_id']][top:bottom,left:right]
            overlay[active]=(overlay[active]*.5+np.array([40,220,180])*.5).astype(np.uint8)
            plain=Image.fromarray(base).resize(((right-left)*8,(bottom-top)*8),Image.Resampling.NEAREST)
            marked=Image.fromarray(overlay).resize(plain.size,Image.Resampling.NEAREST);draw=ImageDraw.Draw(marked)
            bx1,by1,bx2,by2=port['bbox_xyxy'];draw.rectangle(((bx1-left)*8,(by1-top)*8,(bx2-left)*8-1,(by2-top)*8-1),outline='orange',width=2)
            pts=[((x-left)*8+4,(y-top)*8+4) for x,y in candidate['path_xy']]
            if len(pts)>1:draw.line(pts,fill='magenta',width=2)
            sheet=Image.new('RGB',(plain.width*2,plain.height));sheet.paste(plain,(0,0));sheet.paste(marked,(plain.width,0))
            sheet.save(OUT/f'{name}_{port["id"]}_{candidate["record_id"]}.png')
    usage=Counter(c['record_id'] for p in audits for c in p['candidates'])
    for p in audits:
        p['shared_record_ids']=[c['record_id'] for c in p['candidates'] if usage[c['record_id']]>1]
        if p['shared_record_ids'] and len(p['candidates'])==1:
            p['state']='shared_segment_across_ports_ambiguous'
    assert hashlib.sha256(map_path.read_bytes()).hexdigest()==map_digest
    result=dict(image_binding=binding,entry_map_sha256=map_digest,roi_unchanged=True,
                hsv_gate=dict(h=[90,135],s=[80,255],v=[30,255]),neighbourhood=4,
                gap_filling_used=False,ports=audits,connection_edges=[],independent_ground_truth=False)
    result['directional_mode']=args.directional
    result['direction_declaration']='source-review experiment: six lower-entry drafts face down; unconfirmed' if args.directional else None
    (OUT/f'{name}_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    summaries.append(dict(case=name,ports=[dict(port_id=p['port_id'],state=p['state'],candidate_records=[c['record_id'] for c in p['candidates']]) for p in audits]))
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
print(json.dumps(summaries,indent=2))
files=sorted(OUT.glob('cabinet_*_source_entry_*.png'))
sheet=Image.new('RGB',(1200,((len(files)+1)//2)*430),'#eef0f2');draw=ImageDraw.Draw(sheet)
for i,file in enumerate(files):
    with Image.open(file) as image:
        tile=image.convert('RGB');tile.thumbnail((590,400))
    x,y=(i%2)*600,(i//2)*430
    draw.text((x+4,y+3),file.stem,fill='black');sheet.paste(tile,(x+4,y+25))
sheet.save(OUT/'review_sheet.png')
