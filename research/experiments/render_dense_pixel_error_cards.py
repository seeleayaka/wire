"""Native crop error evidence; green=GT, orange=fixed failed head predictions."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import read_image,sha


def main():
    import cv2
    import torch
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    errorfolder=ROOT/'artifacts/dense_pixel_error_audit_20261003'
    report=json.loads((errorfolder/'report.json').read_text(encoding='utf-8'))
    dataset=ROOT/'artifacts/port_training_multiscale_20261002/dataset'
    records=[r for r in json.loads((dataset/'dataset_manifest.json').read_text(encoding='utf-8'))['records'] if r['split']=='val']
    candidates=sorted(report['unmatched'],key=lambda r:-r['prediction']['confidence'])
    selections=[];seen=set()
    for row in candidates:
        if row['source_image'] in seen:continue
        selections.append(row);seen.add(row['source_image'])
        if len(selections)==6:break
    canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for i,row in enumerate(selections):
        record=records[row['crop_index']];path=dataset/record['image'];assert sha(path)==record['image_sha256']
        image=read_image(path);h,w=image.shape[:2];box=row['prediction']['box_xyxy'];l,t,r,b=box
        side=max(180,min(400,int(max(r-l,b-t)*4)));x=max(0,min(w-side,int((l+r-side)/2)));y=max(0,min(h-side,int((t+b-side)/2)))
        item=torch.load(ROOT/'artifacts/dense_pixel_port_probe_20261003/full/features/inner_val'/f"{row['crop_index']:04d}.pt",map_location='cpu',weights_only=True)
        for target in item['boxes']:
            ll,tt,rr,bb=map(round,target['box']);cv2.rectangle(image,(ll,tt),(rr,bb),(0,180,0),2)
        ll,tt,rr,bb=map(round,box);cv2.rectangle(image,(ll,tt),(rr,bb),(0,130,255),2)
        patch=image[y:min(h,y+side),x:min(w,x+side)]
        panel=Image.fromarray(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB));panel=panel.resize((380,380))
        xx=(i%3)*400;yy=(i//3)*450
        draw.text((xx+8,yy+3),f"{row['source_image']} / crop{row['crop_index']}",fill='black',font=font)
        draw.text((xx+8,yy+24),f"c{row['prediction']['class_id']} score{row['prediction']['confidence']:.3f} same-IoU{row['best_same_iou']:.3f}",fill='black',font=font)
        canvas.paste(panel,(xx+8,yy+52))
    path=errorfolder/'unmatched_cards.jpg'
    if path.exists():raise FileExistsError('Preserve evidence')
    canvas.save(path,quality=95)
    (errorfolder/'cards.json').write_text(json.dumps(dict(path=str(path),selections=selections,manual_visual_acceptance=False),indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
