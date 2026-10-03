import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
from core_port_resolution_ab_20261002 import DATA,score
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/bounded_port_rescue_real_20261002'
baseline=json.loads((ROOT/'artifacts/rescue_mainline_ab_20261002_v2/report.json').read_text(encoding='utf-8'))
rows=[]
for case in baseline['cases']:
    original=Path(case['output']);report=json.loads((original/'report_off.json').read_text(encoding='utf-8'))
    old=json.loads((original/'rescue_evidence.json').read_text(encoding='utf-8'))
    new=json.loads((OUT/Path(case['image']).stem/'evidence.json').read_text(encoding='utf-8'))
    targets=[];matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
    for line in (DATA/'labels/test01'/(Path(case['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        if cls not in (3,4):continue
        l,t,r,b=(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736
        q=cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]),matrix).reshape(-1,2)
        targets.append(dict(class_id=int(cls)-3,box=[q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max()]))
    def adapt(result):return [dict(class_id=h['box']['class_id'],box_xyxy=[h['box'][k] for k in ('left','top','right','bottom')]) for h in result['rescue_hints']]
    rows.append(dict(image=case['image'],before=score(adapt(old),targets),after=score(adapt(new),targets)))
(OUT/'evaluation.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
