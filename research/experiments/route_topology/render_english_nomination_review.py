"""Source-pixel review crops only; no synthetic repair or semantic annotation."""
from pathlib import Path
import sys
import json
from PIL import Image,ImageDraw,ImageFont
from run_review import read,save
from core import image_binding,sha256

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'artifacts/route_english_mask_ocr_20261005'
OUT=ROOT/'artifacts/route_english_nomination_review_20261005'


def main():
    if OUT.exists():raise FileExistsError('preserve actual pixel review')
    OUT.mkdir();items=[]
    for name in ['cabinet_1','cabinet_2']:
        p=INPUT/name/'report.json';r=read(p)
        assert image_binding(r['image_path'])==r['image_binding']
        with Image.open(r['image_path']) as opened:original=opened.convert('RGB')
        for g in r['fresh']['groups']:
            if not g['high_score_consistent_text_on_mask']:continue
            box=g['text_group']['bbox_xyxy'];center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
            # Same32 native pixels of context, visualization only, not OCR input.
            crop=[max(0,int(center[0])-32),max(0,int(center[1])-32),min(original.width,int(center[0])+33),min(original.height,int(center[1])+33)]
            native=original.crop(crop);zoom=native.resize((native.width*8,native.height*8),Image.Resampling.NEAREST)
            image=Image.new('RGB',(1040,600),'white');image.paste(zoom,(10,70));marked=zoom.copy();draw=ImageDraw.Draw(marked)
            draw.rectangle([(box[0]-crop[0])*8,(box[1]-crop[1])*8,(box[2]-crop[0])*8,(box[3]-crop[1])*8],outline='#087b70',width=2)
            image.paste(marked,(530,70));draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
            draw.text((10,10),name+' actual source pixels; 8x nearest-neighbor. NOT verified identity/connection.',fill='black',font=font)
            draw.text((10,40),'Original unmarked context',fill='black',font=font);draw.text((530,40),'OCR polygon nomination: '+g['text_group']['best_text'],fill='black',font=font)
            path=OUT/(name+'_'+g['text_group']['group_id']+'.png');image.save(path)
            items.append(dict(image=str(path),source_path=r['image_path'],source_sha256=r['image_binding']['image_sha256'],source_crop_xyxy=crop,
                 nominated_text=g['text_group']['best_text'],mask_id=g['nominated_mask_id'],geometry_eligible=g['all_reading_memberships'][0]['supports'][0]['topology_geometry_eligible'],
                 reading_source_ids=g['text_group']['record_ids'],no_interpolated_letter_repair=True,actual_human_confirmation=False,new_confirmed_connections=0))
    save(OUT/'report.json',dict(status='complete',items=items,no_model_inference=True,no_deployment=True,field_accuracy=None))
    print(json.dumps(dict(status='complete',items=items)))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
