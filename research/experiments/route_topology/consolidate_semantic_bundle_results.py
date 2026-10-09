"""Bound evidence rounds without pretending all30 had fresh SAM this round."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from core import sha256
from run_prompt_contrast import save,verify
from run_audit import source_pins
ROOT=Path(__file__).resolve().parents[2]
def main():
    old_path=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_review_20261006/report.json'
    new_path=ROOT/'artifacts/mendeley_semantic_visible_bundle_review_20261006/report.json'
    audit_path=ROOT/'artifacts/mendeley_semantic_visible_bundle_native_audit_20261006/report.json'
    old=json.loads(old_path.read_text(encoding='utf-8'));new=json.loads(new_path.read_text(encoding='utf-8'))
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status']=='PASS' and audit['source_report_sha256']==sha256(new_path)
    verify(old['pins']);before=source_pins()
    updated={c['id']:c for c in new['cases']};results=[];gains=[]
    for previous in old['cases']:
        identity=previous['id'];current=updated.get(identity,previous)
        assert current['observation']['source_binding']==previous['observation']['source_binding']
        if previous['comparison']['decision']!='insufficient_evidence':
            assert current['comparison']['decision']==previous['comparison']['decision']
        elif current['comparison']['decision']!='insufficient_evidence':gains.append(identity)
        results.append(dict(**current,evidence_round='new_fresh_original_pose_DINO_SAM_round' if identity in updated else 'previous_verified_full_batch'))
    assert len(results)==30 and gains==['case_18']
    out=ROOT/'artifacts/mendeley_semantic_bundle_consolidated_20261006';out.mkdir(exist_ok=False)
    counts={d:sum(c['comparison']['decision']==d for c in results) for d in old['decision_counts']}
    save(out/'report.json',dict(status='complete',cases=results,decision_counts=counts,
        previous_counts=old['decision_counts'],new_scoped_bundle_observations=gains,old_supported_losses=[],
        all30_fresh_SAM_in_this_round=False,fresh_SAM_cases_in_this_round=['reference','case_04','case_18'],
        mixture_of_explicitly_bound_evidence_rounds=True,not_blind_or_field_accuracy=True,
        electrical_connections_confirmed=0,production_unchanged=True,deployed=False,
        inputs={str(p):sha256(p) for p in [old_path,new_path,audit_path]}))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    sheet=Image.new('RGB',(1320,620),'#f7f8fa');draw=ImageDraw.Draw(sheet)
    panels=[('参考线束','reference','mask_003','人工确认的可见参考范围'),
        ('新增一致结果','case_18','mask_004','同一原生片段覆盖两端；不比较走线形状'),
        ('仍需复核','case_04','mask_004','两端分属不同片段，不补线、不猜连接')]
    for i,(title,identity,mask,caption) in enumerate(panels):
        x=i*440;draw.text((x+15,12),title,font=font,fill='#263342')
        with Image.open(new_path.parent/(identity+'_cable_plus_reference_anatomy_box')/(mask+'.png')) as photo:
            image=photo.convert('RGB');image.thumbnail((420,460));sheet.paste(image,(x+10,50))
        draw.text((x+10,525),caption,font=small,fill='#263342')
    draw.text((16,575),'实拍研发演示：只能支持可见线束关系；不证明逐芯接法、内部接点或电气导通。',font=small,fill='#775b30')
    sheet.save(out/'visible_bundle_progress.png')
    assert source_pins()==before
    print(json.dumps(counts))
if __name__=='__main__':main()
