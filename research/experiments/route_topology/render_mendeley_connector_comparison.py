"""Three actual original-image observations with explicit inference provenance.

Existing fresh positive SAM runs are explicitly reused for comparison. Only the
negative SAM run is new here. Never label this sheet a single fresh full-chain run.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from core import sha256
from run_paired_evidence import load_run
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    positive=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    negative=ROOT/'artifacts/mendeley_exposed_socket_sam_20261005'
    preview=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_20261005'
    pos=json.loads((positive/'protocol.json').read_text(encoding='utf-8'))
    neg=json.loads((negative/'protocol.json').read_text(encoding='utf-8'))
    cases=json.loads((preview/'report.json').read_text(encoding='utf-8'))['cases']
    rows=[(pos['cases'][0],positive/'reference','正常参考（待确认）',None),
        (pos['cases'][1],positive/'inspection','走线变化：插头仍可见',next(r for r in cases if r['id']=='case_11')),
        (neg['cases'][0],negative/'case_06','插座接点露出',next(r for r in cases if r['id']=='case_06'))]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    output=ROOT/'artifacts/mendeley_connector_comparison_20261005';output.mkdir(exist_ok=False)
    sheet=Image.new('RGB',(1410,760),'#edf0f3');draw=ImageDraw.Draw(sheet)
    draw.text((14,10),'可见插接状态对照：不把走线变化直接判成拔线',font=font,fill='#263342')
    draw.text((14,44),'正例复用今日已完成原图SAM；空插座为新增原图SAM。尚未通过完整拓扑／电气验收。',font=small,fill='#775c31')
    records=[]
    for i,(case,run,title,row) in enumerate(rows):
        original=case['original_source']['path']
        if sha256(original)!=case['original_source']['image_sha256']:raise ValueError('original drift')
        with Image.open(original) as im:rgb=np.asarray(im.convert('RGB'));crop=rgb[case['crop_box_xyxy'][1]:case['crop_box_xyxy'][3],case['crop_box_xyxy'][0]:case['crop_box_xyxy'][2]].copy()
        with Image.open(case['source']['path']) as im:saved=np.asarray(im.convert('RGB'))
        if not np.array_equal(crop,saved):raise ValueError('SAM input not original pixels')
        origin,mask_records=load_run(run/'cable_plus_reference_anatomy_box')
        # Fixed third mask of each anatomy recipe, kept only as unaccepted bundle diagnostic.
        mask_path=Path(next(r['source_mask_path'] for r in mask_records if r['record_id']=='mask_003'))
        with Image.open(mask_path) as im:active=np.asarray(im.convert('L'))>0
        if active.shape!=crop.shape[:2]:raise ValueError('mask/crop size differs')
        overlay=crop.astype(float);overlay[active]=overlay[active]*.72+np.array([239,160,50])*.28
        x=i*470;draw.rectangle((x+8,86,x+462,748),fill='white')
        draw.text((x+20,100),title,font=font,fill='#2b6c64' if i<2 else '#b65337')
        photo=Image.fromarray(overlay.astype(np.uint8));photo.thumbnail((430,430));sheet.paste(photo,(x+20,145))
        if row is not None:
            with Image.open(row['patch_path']) as im:socket=im.convert('RGB')
        else:socket=Image.fromarray(rgb[1000:1050,1600:1700])
        socket=socket.resize((400,100),Image.Resampling.NEAREST);sheet.paste(socket,(x+30,579))
        draw.text((x+20,690),'橙色：线束候选，未确认为单根导线',font=small,fill='#775c31')
        draw.text((x+20,718),'完整关系结论：证据不足',font=small,fill='#775c31')
        records.append({'original':original,'original_sha256':case['original_source']['image_sha256'],
            'sam_run':str(run),'mask_path':str(mask_path),'mask_sha256':sha256(mask_path),
            'positive_runs_explicitly_reused':i<2,'new_negative_run':i==2,'topology_decision':'insufficient_evidence'})
    sheet.save(output/'connector_comparison.png')
    save(output/'report.json',{'status':'complete','rows':records,'single_fresh_full_chain_run':False,
        'three_distinct_original_photos':True,'reference_review_confirmed':False,
        'new_confirmed_connections':0,'not_accuracy_test':True,
        'pins':{str(p):sha256(p) for p in [Path(__file__),positive/'protocol.json',negative/'protocol.json',preview/'report.json']}})
    print(output/'connector_comparison.png')


if __name__=='__main__':main()
