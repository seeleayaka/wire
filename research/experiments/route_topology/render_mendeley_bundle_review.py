"""Evidence sheet: whole raw bundle masks and reference-only anchor proposals."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core import sha256
from run_paired_evidence import load_run
from run_prompt_contrast import verify
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'));verify(protocol['pins'])
    audit_path=ROOT/'artifacts/mendeley_bundle_support_audit_20261005/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    pose_path=ROOT/'artifacts/mendeley_local_anchor_pose_20261005/report.json'
    poses=json.loads(pose_path.read_text(encoding='utf-8'))['anchors']
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    registration_path=ROOT/'artifacts/mendeley_visible_fan_scope_20261005/registration/report.json'
    registration=json.loads(registration_path.read_text(encoding='utf-8'))['registration']
    h=np.asarray(registration['source_to_reference_homography'])
    output=ROOT/'artifacts/mendeley_bundle_review_sheet_20261005'
    if output.exists():raise FileExistsError('preserve previous evidence')
    output.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    big=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    canvas=Image.new('RGB',(1080,770),'#f5f6f8');draw=ImageDraw.Draw(canvas)
    draw.text((24,12),'CPU 風扇線束：原圖重跑後的可見證據',font=big,fill='#25364a')
    draw.text((24,46),'判定：證據不足／未驗收。橙色為完整分割輸出，綠框為參考標記草稿。',font=font,fill='#926113')
    selections=[]
    for index,case in enumerate(protocol['cases']):
        side=case['id'];rows=next(c['records'] for c in audit['cases'] if c['side']==side)
        selected=[r for r in rows if r['score']>=.75 and not r['boundary_truncated']
            and r['component_has_both_anchor_support']]
        if len(selected)!=1:raise ValueError('do not cherry pick ambiguous bundle candidates')
        row=selected[0];origin,records=load_run(source/side/'cable_plus_reference_anatomy_box')
        record=next(r for r in records if r['record_id']==row['record_id'])
        with Image.open(origin['image_path']) as opened:rgb=np.asarray(opened.convert('RGB'))
        with Image.open(record['source_mask_path']) as opened:active=np.asarray(opened.convert('L'))>0
        rgb=rgb.astype(float);rgb[active]=rgb[active]*.72+np.array([242,159,37])*.28
        photo=Image.fromarray(rgb.astype('uint8'));pd=ImageDraw.Draw(photo)
        transform=np.eye(3) if side=='reference' else np.linalg.inv(h)
        offset=np.array(case['crop_box_xyxy'][:2]);anchor_polygons=[]
        for anchor in scope['anchors']:
            a,b,c,d=anchor['bbox_xyxy'];points=np.array([[a,b,1],[c,b,1],[c,d,1],[a,d,1]],float)@transform.T
            polygon=points[:,:2]/points[:,2:]-offset
            pd.line([tuple(v) for v in polygon]+[tuple(polygon[0])],fill='#07856b',width=2)
            pd.text(tuple(polygon[0]+[0,-19]),anchor['id'],font=font,fill='#007b64')
            anchor_polygons.append({'id':anchor['id'],'crop_polygon_xy':polygon.tolist()})
        if side=='inspection':
            for pose in poses:
                if pose['reliable_localization']:continue
                points=np.asarray(pose['inspection_anchor_polygon_xy'])-offset
                pd.line([tuple(v) for v in points]+[tuple(points[0])],fill='#c2437f',width=2)
        photo=photo.resize((500,500),Image.Resampling.NEAREST);x=24+index*532
        canvas.paste(photo,(x,116));draw=ImageDraw.Draw(canvas)
        draw.text((x,86),'參考原图' if side=='reference' else '待檢原图（自動對位）',font=big,fill='#25364a')
        draw.text((x,626),f"SAM 分數 {row['score']:.3f}；完整保留 {row['raw_foreground_component_count']} 個像素區塊",font=font,fill='#27384b')
        draw.text((x,652),'有同一區塊觸及兩個標記區；這不是電氣連通證明。',font=font,fill='#926113')
        selections.append({'side':side,'record_id':row['record_id'],'mask_array_sha256':row['raw_mask_array_sha256'],
            'original_source':case['original_source'],'anchors':anchor_polygons,'qualitative_only':True})
    draw.text((24,695),'尚缺：參考標記正式復核、風扇出線位置驗證、線束身份與分叉／遮擋消歧。',font=font,fill='#926113')
    draw.text((24,726),'粉框：局部定位與全局定位有偏差，未因候選線束得分高就放行。',font=font,fill='#6c5362')
    canvas.save(output/'bundle_review.png')
    save(output/'report.json',{'status':'complete','selections':selections,'decision':'insufficient_evidence',
        'reference_review_confirmed':False,'new_confirmed_connections':0,'deployed':False,
        'source_pins':{str(p):sha256(p) for p in [Path(__file__),source/'protocol.json',audit_path,pose_path,
            scope_path,registration_path]},'pixel_masks_not_edited':True,'only_display_resized':True})
    print(output/'bundle_review.png')


if __name__=='__main__':main()
