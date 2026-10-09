"""Exact fresh patch binding, preservation checks, and actual-crop review sheet."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from core import sha256
from run_prompt_contrast import save
from run_audit import source_pins
ROOT=Path(__file__).resolve().parents[2]
def main():
    source=ROOT/'artifacts/mendeley_semantic_visible_fresh_controls_20261006'
    rp=source/'report.json';r=json.loads(rp.read_text(encoding='utf-8'))
    p=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    assert r['status']=='complete' and r['source_controls_passed']
    assert source_pins()==p['production_pins']
    assert all(sha256(f)==d for f,d in p['pins'].items())
    out=ROOT/'artifacts/mendeley_semantic_visible_fresh_audit_20261006';out.mkdir(exist_ok=False)
    new=[];controls=0;demo=[];sheet=Image.new('RGB',(600,480),'white');draw=ImageDraw.Draw(sheet)
    for case in r['cases']:
        assert sha256(case['source']['path'])==case['source']['image_sha256']
        rgb=np.asarray(Image.open(case['source']['path']).convert('RGB'))
        h=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(case['pose']['inspection_to_reference_local'])
        actual=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
        saved=np.asarray(Image.open(source/(case['dataset']+'_'+case['id']+'_socket.png')).convert('RGB'))
        assert np.array_equal(actual,saved)
        pred=case['prediction'];prior=pred['prior_compound_candidate']
        if prior is not None:assert prior==pred['visual_label_candidate']
        if case['dataset']=='source_controls':
            assert pred['visual_label_candidate']==case['expected'];controls+=1
        else:demo.append(case)
        if pred['semantic_visible_rescue']:
            assert prior is None and pred['visual_label_candidate']==1 and pred['semantic_visible_outer_labels']==[1]*9
            y=len(new)*120;draw.text((10,y+4),case['id']+' / actual newly inferred socket crop',fill='black')
            sheet.paste(Image.fromarray(actual).resize((200,100)),(0,y+20))
            photo=Image.fromarray(rgb);photo.thumbnail((170,110));sheet.paste(photo,(250,y+10))
            new.append(case['id'])
    assert controls==5 and len(demo)==30 and new==r['new_visible_candidates']
    assert source_pins()==p['production_pins']
    sheet.save(out/'new_visible_socket_review.png')
    save(out/'report.json',dict(status='PASS',source_report_sha256=sha256(rp),
        exact_original_warp_bytes_checked=35,source_controls_passed=5,
        new_visible_candidates=new,old_losses=0,visual_semantic_review_pending=True,
        not_independent_reinference_or_field_accuracy=True,production_unchanged=True,deployed=False))
    print(json.dumps(dict(status='PASS',new_visible_candidates=new)))
if __name__=='__main__':main()
