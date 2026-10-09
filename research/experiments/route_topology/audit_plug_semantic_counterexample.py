"""Compare fixed positive/negative source masks without changing topology policy."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from run_prompt_contrast import digest, save
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]

def main():
    base=ROOT/'artifacts/mendeley_plug_source_probe_v2_20261006'
    negative=ROOT/'artifacts/mendeley_plug_negative_diagnostic_20261006'
    assert json.loads((negative/'report.json').read_text(encoding='utf-8'))['status']=='complete'
    out=ROOT/'artifacts/mendeley_plug_semantic_counterexample_audit_20261006'
    out.mkdir(exist_ok=False)
    pins=source_pins()
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    sheet=Image.new('RGB',(800,680),'white');draw=ImageDraw.Draw(sheet)
    rows=[]
    for row,(identity,folder) in enumerate([('reference',base),('source_exposed_01',negative)]):
        case=next(c for c in protocol['cases'] if c['id']==identity)
        run=folder/identity/'plug_plus_socket_anatomy_box'
        manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
        for filename,expected in manifest['verified_files'].items():assert digest(filename)==expected
        assert manifest['image_binding']['image_sha256']==case['source']['image_sha256']
        assert digest(case['original_source']['path'])==case['original_source']['image_sha256']
        crop=np.asarray(Image.open(case['original_source']['path']).convert('RGB').crop(case['crop_box_xyxy']))
        assert np.array_equal(crop,np.asarray(Image.open(case['source']['path']).convert('RGB')))
        raw_report=json.loads((run/'sam/report.json').read_text(encoding='utf-8'))
        union=np.asarray(Image.open(run/'sam/mask_union.png').convert('L'))>0
        l,t,r,b=np.asarray(case['positive_box_source_xyxy'])-np.asarray(case['crop_box_xyxy'][:2]*2)
        overlay=crop.copy()
        overlay[union]=(.55*crop[union]+.45*np.array([255,40,70])).astype(np.uint8)
        y=340*row
        draw.text((5,y+10),identity+' / actual original crop',fill='black')
        draw.text((405,y+10),'Unchanged plug+box mask (not occupancy proof)',fill='black')
        sheet.paste(Image.fromarray(crop).resize((400,300)),(0,y+35))
        sheet.paste(Image.fromarray(overlay).resize((400,300)),(400,y+35))
        rows.append(dict(id=identity,instance_count=raw_report['instance_count'],scores=raw_report['scores'],
             native_socket_box_mask_pixels=int(union[t:b,l:r].sum()),
             original_sha256=case['original_source']['image_sha256'],
             mask_union_sha256=digest(run/'sam/mask_union.png')))
    sheet.save(out/'positive_negative_overlay.png')
    assert source_pins()==pins
    save(out/'report.json',dict(status='complete',cases=rows,
         fresh_original_crop_byte_checks=True,manifest_files_verified=True,
         reference_evidence_explicit_reanalysis=True,new_inference=False,
         production_unchanged=True,semantic_review_pending=True,
         score_is_not_connector_identity=True,topology_policy_unchanged=True,deployed=False))
    print(json.dumps(rows))

if __name__=='__main__':main()
