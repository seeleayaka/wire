"""Display every eligible old mask's tips in wider original context.

Display margins are NOT port ROIs or acceptance parameters. No labels/netlist
are created. Evidence only, and no inference or mainline mutation.
"""
from pathlib import Path
import sys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_paired_evidence import load_run
from run_review import save
from core import sha256

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'artifacts/route_endpoint_context_20261005'
    out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for name,relative in [('full_cabinet_1','bound_sam_topology_20261001/cabinet_1_fresh'),
                           ('crop_cabinet_1','sam_crop_coverage_20261001/cabinet_1_crop_run'),
                           ('crop_cabinet_2','sam_crop_coverage_20261001/cabinet_2_crop_run')]:
        origin,records=load_run(ROOT/'artifacts'/relative)
        eligible=[r for r in records if r['score']>=.75 and not r['boundary_truncated'] and
                  r['geometry']['state']=='simple_visible_path']
        with Image.open(origin['image_path']) as opened:
            image=opened.convert('RGB')
        # Wider than line tips for manual scene context only.
        radius=max(8,int(math.ceil(math.hypot(*image.size)*.04)))
        sheet=Image.new('RGB',(950,max(1,len(eligible))*180),'#f7f8fa')
        font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
        entries=[]
        for row,record in enumerate(eligible):
            draw=ImageDraw.Draw(sheet)
            draw.text((8,row*180+5),f"{record['record_id']} score={record['score']:.3f} / NOT a confirmed connection",font=font,fill='black')
            source=Path(record['source_mask_path'])
            with Image.open(source) as opened:
                mask=np.asarray(opened.convert('L'))>0
            for index,(x,y) in enumerate(record['geometry']['tips_xy']):
                box=(max(0,x-radius),max(0,y-radius),min(image.width,x+radius+1),min(image.height,y+radius+1))
                tile=image.crop(box)
                d=ImageDraw.Draw(tile);d.ellipse((x-box[0]-2,y-box[1]-2,x-box[0]+2,y-box[1]+2),outline='red',width=1)
                tile=tile.resize((145,145))
                sheet.paste(tile,(8+index*155,row*180+28))
            pixels=np.asarray(image,dtype=np.float32).copy()
            pixels[mask]=.35*pixels[mask]+.65*np.array([30,180,215])
            full=Image.fromarray(pixels.astype(np.uint8))
            full.thumbnail((280,145))
            sheet.paste(full,(330,row*180+28))
            d=ImageDraw.Draw(sheet)
            d.text((640,row*180+40),'Tip 1 / Tip 2 / whole source',font=font,fill='#554433')
            d.text((640,row*180+65),'port identity: unknown',font=font,fill='#554433')
            entries.append({'record_id':record['record_id'],'score':record['score'],
                            'tips_xy':record['geometry']['tips_xy'],'mask_sha256':sha256(source),
                            'actual_wire_entry_confirmed':False,'expected_connection':None})
        sheet.save(out/f'{name}.png')
        cases.append({'case':name,'origin':origin,'eligible_records':entries})
    save(out/'report.json',{'status':'complete','source':cases,
            'display_radius_diagonal_ratio':.04,'display_margin_not_acceptance_threshold':True,
            'new_inference':False,'manual_audit':'pending','new_confirmed_connections':0})
    print({'eligible_per_case':[len(c['eligible_records']) for c in cases]})


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
