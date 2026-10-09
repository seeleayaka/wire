"""All120 normal TRAIN socket source patches for actual visual label review."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from core import sha256
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_normal_appearance_20261005'
    old=json.loads((source/'report.json').read_text(encoding='utf-8'))
    inputs=[r for r in old['prepared'] if r['phase'] in ['fit','calibration']]
    if len(inputs)!=120 or len({r['path'] for r in inputs})!=120:raise ValueError('all120 source inputs required')
    output=ROOT/'artifacts/mendeley_normal_source_socket_review_20261005'
    if output.exists():raise FileExistsError('preserve existing review')
    output.mkdir(exist_ok=False);rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for index,row in enumerate(inputs):
        if sha256(row['path'])!=row['input_sha256']:raise ValueError('original source drift')
        with Image.open(row['path']) as opened:rgb=np.asarray(opened.convert('RGB'))
        matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(row['pose']['inspection_to_reference_local'])
        patch=cv2.warpPerspective(rgb,matrix,(100,50),flags=cv2.INTER_LINEAR)
        identity=f'normal_source_{index+1:03d}';destination=output/(identity+'_socket.png')
        Image.fromarray(patch).save(destination)
        if index%20==0:page=Image.new('RGB',(1800,840),'#f4f5f7')
        cell=Image.new('RGB',(360,210),'white');draw=ImageDraw.Draw(cell)
        draw.text((7,4),identity,font=font,fill='#25364a');draw.text((7,25),'正常来源；仍须核查可见插头',font=font,fill='#926113')
        cell.paste(Image.fromarray(patch).resize((340,170),Image.Resampling.NEAREST),(10,40))
        page.paste(cell,((index%5)*360,((index%20)//5)*210))
        if index%20==19:page.save(output/(f'normal_source_page_{index//20+1}.png'))
        rows.append({'id':identity,'source_path':row['path'],'source_sha256':row['input_sha256'],
            'patch_path':str(destination),'patch_sha256':sha256(destination),'pose':row['pose'],
            'occupancy_label':None,'source_pool':'known_normal_source','human_reviewed':False})
    save(output/'review_inventory.json',{'status':'prepared','rows':rows,'inspection_inputs_read':False,
        'occupancy_labels_confirmed':0,'reuses_verified_source_poses':True,'fresh_source_RGB_decode':True,
        'source_report_sha256':sha256(source/'report.json'),
        'source_pins':{str(p):sha256(p) for p in [Path(__file__),source/'report.json']}})
    print(output)


if __name__=='__main__':main()
