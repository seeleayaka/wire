"""All30 visible socket evidence, without promoting appearance to wiring verdict."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from core import sha256
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    if report['status']!='complete' or report['protocol_sha256']!=sha256(source/'protocol.json'):raise ValueError('incomplete/drifted report')
    if any(sha256(p)!=d for p,d in protocol['pins'].items()):raise ValueError('input/model/source changed')
    output=ROOT/'artifacts/mendeley_socket_phenotype_review_20261005';output.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    names={'mating_body_visible':'插头本体可见（候选）','socket_contacts_exposed':'插座接点露出（候选）','uncertain':'暂不判断'}
    colors={'mating_body_visible':'#27756b','socket_contacts_exposed':'#bd573b','uncertain':'#8b722c'}
    for page in range(2):
        sheet=Image.new('RGB',(1320,1570),'#edf0f3');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),'全部30张原图重跑：只看插头可见状态，未确认接线或电气导通',font=font,fill='#465467')
        for j,r in enumerate(report['cases'][page*15:(page+1)*15]):
            cell=Image.new('RGB',(440,300),'white');d=ImageDraw.Draw(cell)
            d.text((10,6),r['id']+'  '+names[r['phenotype']],font=font,fill=colors[r['phenotype']])
            d.text((10,35),'正式拓扑结论仍为：证据不足',font=small,fill='#775c31')
            if 'patch_path' in r:
                if sha256(r['patch_path'])!=r['patch_sha256']:raise ValueError('render patch changed')
                with Image.open(r['patch_path']) as im:patch=im.convert('RGB').resize((400,200),Image.Resampling.NEAREST)
                cell.paste(patch,(20,63))
            else:d.text((12,110),'定位不通过，不套用旧坐标',font=font,fill='#8b722c')
            d.text((10,274),'颜色／边缘不一致就保留为不确定',font=small,fill='#5b6777')
            cell.save(output/(r['id']+'.png'));sheet.paste(cell,((j%3)*440,50+(j//3)*300))
        sheet.save(output/f'phenotype_page_{page+1}.png')
    save(output/'report.json',{'status':'complete','all30_included':True,'visible_phenotype_only':True,
        'new_confirmed_connections':0,'pins':{str(p):sha256(p) for p in [Path(__file__),source/'report.json',source/'protocol.json']}})
    print(str(output))


if __name__=='__main__':main()
