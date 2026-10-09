"""All30 raw-photo localization proposals, including failures; no cable verdicts."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from core import sha256
from prepare_mendeley_scope import inspection_scope
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_heldout_pose_all30_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if report['status']!='complete' or report['protocol_sha256']!=sha256(source/'protocol.json'):
        raise ValueError('batch evidence incomplete or changed')
    for path,digest in protocol['pins'].items():
        if sha256(path)!=digest:raise ValueError('batch pinned file changed')
    output=ROOT/'artifacts/mendeley_pose_all30_review_20261005'
    if output.exists():raise FileExistsError('preserve earlier review')
    output.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    sheet=Image.new('RGB',(1800,2460),'#eef0f4');draw=ImageDraw.Draw(sheet)
    draw.text((12,12),'全部30张原图端点定位候选：不代表线束连接正确，不是识别准确率',font=font,fill='#8c6418')
    rows=[]
    for index,case in enumerate(report['cases']):
        path=Path(case['input_path'])
        if sha256(path)!=case['input_sha256']:raise ValueError('source image drift')
        cell=Image.new('RGB',(360,402),'white');cd=ImageDraw.Draw(cell)
        supported=len(case['anchors'])==2 and all(a['localization_proposal_supported'] for a in case['anchors'])
        cd.text((6,5),case['id']+' | '+('两处位置候选有支持' if supported else '局部定位证据不足'),font=font,
            fill='#247663' if supported else '#af632a')
        cd.text((6,24),'未确认端点身份／未判接线',font=font,fill='#8b641b')
        registration=case['registration'];box=None
        if registration['alignment_quality']['reliable']:
            with Image.open(path) as opened:
                image=opened.convert('RGB');box=inspection_scope([1380,870,1780,1270],
                    np.asarray(registration['source_to_reference_homography']),image.size)
                crop=image.crop(box)
            cropdraw=ImageDraw.Draw(crop)
            for anchor in case['anchors']:
                if 'inspection_anchor_polygon_xy' not in anchor:continue
                polygon=np.asarray(anchor['inspection_anchor_polygon_xy'])-np.array(box[:2])
                color='#168a72' if anchor['localization_proposal_supported'] else '#c27035'
                cropdraw.line([tuple(p) for p in polygon]+[tuple(polygon[0])],fill=color,width=2)
            crop.thumbnail((344,344));cell.paste(crop,(8,49))
        else:cd.text((8,90),'全局对位失败：不套用坐标',font=font,fill='#af632a')
        sheet.paste(cell,((index%5)*360,48+(index//5)*402));cell.save(output/(case['id']+'.png'))
        rows.append({'id':case['id'],'source_path':str(path),'source_sha256':case['input_sha256'],
            'crop_xyxy':box,'spatial_proposal_supported':supported,'connection_verdict':'insufficient_evidence'})
    sheet.save(output/'all30_pose_proposals.png')
    save(output/'report.json',{'status':'complete','all30_included':True,'rows':rows,
        'new_confirmed_connections':0,'connection_verdicts_rendered_as_normal':0,
        'pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('prepare_mendeley_scope.py'),
            source/'protocol.json',source/'report.json']}})
    print(output/'all30_pose_proposals.png')


if __name__=='__main__':main()
