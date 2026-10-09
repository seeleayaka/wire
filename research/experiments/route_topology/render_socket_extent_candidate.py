"""Render only: show two changed development candidates and original raw masks."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_review import read,verified_run
from run_prompt_contrast import verify

ROOT=Path(__file__).resolve().parents[2]


def main():
    output=ROOT/'artifacts/mendeley_socket_extent_candidate_20261006'
    report=read(output/'report.json');verify(report['pins'])
    evidence=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    prep=read(evidence/'preparation_report.json');font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    sheet=Image.new('RGB',(1100,700),'#f3f5f7');draw=ImageDraw.Draw(sheet)
    draw.text((20,15),'碎点冲突修正：开发集候选，不是新 SAM 或电气准确率',font=font,fill='#273645')
    for i,identity in enumerate(report['changed_cases']):
        row=next(r for r in prep['cases'] if r['id']==identity)
        result=next(r for r in report['cases'] if r['id']==identity)
        case=row['crop_context'];run=evidence/identity/'cable_plus_reference_anatomy_box'
        manifest=read(run/'run_manifest.json');native=verified_run(run,manifest['image_binding']['image_path'])
        with Image.open(case['source']['path']) as im:photo=np.asarray(im.convert('RGB')).copy()
        active=np.zeros(photo.shape[:2],bool)
        for mask,score in zip(native['paths'],native['scores']):
            if score<.75:continue
            with Image.open(mask) as im:active |= np.asarray(im.convert('L'))>0
        overlay=photo.astype(float);overlay[active]=overlay[active]*.60+np.array([226,148,60])*.40
        x=20+i*540;draw.rectangle((x,60,x+520,680),fill='white')
        draw.text((x+16,75),identity+'  原生高分遮罩（橙色）',font=font,fill='#273645')
        image=Image.fromarray(overlay.astype(np.uint8));image.thumbnail((490,450));sheet.paste(image,(x+16,115))
        draw.text((x+16,585),'原判断：任何碎点碰到插座 → 证据冲突',font=small,fill='#8a7130')
        draw.text((x+16,613),'新候选：插座接点露出，无原生出线段反证',font=small,fill='#a34a31')
        draw.text((x+16,641),'未删除碎点／未拼接线段／来源验证进行中',font=small,fill='#576370')
        if result['observation']['native_outgoing_component_at_socket']:raise ValueError('changed candidate unexpectedly outgoing')
    verify(report['pins']);sheet.save(output/'comparison.png')
    print('Candidate comparison rendered; underlying raw masks unchanged')


if __name__=='__main__':main()
