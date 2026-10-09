"""Fixed PCA diagnostic views + new local OCR; no authenticated topology."""
from pathlib import Path
import sys
import time
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mask_axis_views import axis_transform,warp_context,map_to_source
from run_mask_label_ocr import prepare_cases,valid_row,summarize,OUT as BASELINE
from label_attachment import nominate,mask_context_crop
from identity_ocr import inverse_rotation
from run_identity_ocr import DEPENDENCIES
from run_audit import source_pins
from run_review import save,read
from core import image_binding,sha256

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/route_mask_axis_ocr_20261005'
PLAN=ROOT/'artifacts/route_mask_axis_ocr_preregistration_20261005/PLAN.md'


def main():
    if OUT.exists():raise FileExistsError('preserve diagnostic view experiment')
    cases=prepare_cases();old_protocol=read(BASELINE/'protocol.json')
    assert read(BASELINE/'report.json')['status']=='complete'
    assert all(sha256(p)==d for p,d in {**old_protocol['pins'],**old_protocol['models']}.items())
    before=source_pins();models=old_protocol['models']
    files=[Path(__file__),PLAN,BASELINE/'report.json',BASELINE/'protocol.json',
           *[Path(__file__).with_name(n+'.py') for n in ('mask_axis_views','run_mask_label_ocr','label_attachment','identity_ocr','run_audit','run_paired_evidence','core')]]
    pins={str(p):sha256(p) for p in files}
    for case in cases:
        p=BASELINE/case['id']/'report.json';pins[str(p)]=sha256(p)
    OUT.mkdir();save(OUT/'protocol.json',dict(pins=pins,models=models,mainline_pins=before,plan_sha256=sha256(PLAN),
        no_GT_or_token_repair=True,fresh_OCR=True,fresh_SAM=False,old_pixels_and_masks_pinned_by=str(BASELINE/'protocol.json'),
        min_anisotropy=2,quarters=[0,1,2,3],scale=2,crop_boundary_margin=2,one_observer_each=True,no_deployment=True))
    sys.path.insert(0,str(DEPENDENCIES))
    from rapidocr_onnxruntime import RapidOCR
    engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=2)
    start=time.monotonic();calls=0;totals=[]
    try:
        for case in cases:
            folder=OUT/case['id'];folder.mkdir();source=case['full']['image_path'];binding=case['full']['frame_binding']
            assert image_binding(source)==binding
            with Image.open(source) as opened:image=opened.convert('RGB');pixels=np.asarray(image)
            prior=read(BASELINE/case['id']/'report.json');rows=[];reject=[];transforms=[]
            old_reject={r['record_id'] for r in prior['baseline_analysis_rejections']}
            old=[dict(r,record_id='base_'+r['record_id']) for r in prior['baseline_raw_readings_preserved'] if 'base_'+r['record_id'] not in old_reject]+prior['fresh_readings']
            for mask in case['masks']:
                crop=mask_context_crop(mask['pixels'],image.size)
                transform=axis_transform(mask['pixels'],crop) if crop is not None else dict(state='empty_mask')
                transforms.append(dict(mask_id=mask['id'],**transform))
                if transform['state']!='oriented_diagnostic_view':continue
                oriented=Image.fromarray(warp_context(pixels,crop,transform))
                doubled=np.asarray(oriented.resize((oriented.width*2,oriented.height*2),Image.Resampling.LANCZOS))
                for turn in range(4):
                    save(OUT/'progress.json',dict(status='running',case=case['id'],mask=mask['id'],rotation=turn,calls_completed=calls,seconds=time.monotonic()-start))
                    result,elapsed=engine(np.ascontiguousarray(np.rot90(doubled,turn)[:,:,::-1]));calls+=1
                    view=mask['id']+f'_v{turn}'
                    save(folder/(view+'_raw.json'),dict(rows=result,elapsed=elapsed,transform=transform,quarter_turns=turn))
                    for i,(polygon,text,score) in enumerate(result or []):
                        rid='axis_'+view+f'_{i:03d}'
                        normalized=inverse_rotation(polygon,oriented.width*2,oriented.height*2,turn,2)
                        p=map_to_source(normalized,transform)
                        box=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
                        if box[0]<=crop[0]+2 or box[1]<=crop[1]+2 or box[2]>=crop[2]-2 or box[3]>=crop[3]-2:
                            reject.append(dict(record_id=rid,text=text,score=float(score),reason='text_touches_native_context_boundary'));continue
                        row=dict(record_id=rid,text=text,score=float(score),polygon_source_xy=p.tolist(),bbox_xyxy=box,
                                 view_quarter_turns=turn,crop_proposer_mask_id=mask['id'],source_crop_xyxy=crop,evidence_origin='fresh_whole_mask_PCA_OCR')
                        try:valid_row(row,image.size)
                        except ValueError as exc:
                            reject.append(dict(record_id=rid,text=text,score=float(score),reason=str(exc)));continue
                        rows.append(row)
            fresh=nominate(rows,case['masks'],image.size);combined=nominate(old+rows,case['masks'],image.size)
            save(folder/'report.json',dict(image_path=source,image_binding=binding,axis_transforms=transforms,
                old_raw_readings_preserved=old,fresh_readings=rows,rejections=reject,fresh=fresh,combined=combined,
                baseline_combined_summary=summarize(prior['combined']),no_automatic_identity=True,new_confirmed_connections=0))
            canvas=image.resize((image.width*2,image.height*2),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(canvas)
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
            for g in fresh['groups']:
                text=g['text_group'];box=[v*2 for v in text['bbox_xyxy']];color='#087b70' if g['high_score_consistent_text_on_mask'] else '#b76b00'
                draw.rectangle(box,outline=color,width=2);draw.text((box[0],max(0,box[1]-18)),f"{text['best_text']} => {g['nominated_mask_id'] or '?'}",font=font,fill=color,stroke_width=1,stroke_fill='white')
            canvas.save(folder/'fresh_axis_text.png')
            totals.append(dict(case=case['id'],source_masks=len(case['masks']),oriented_masks=sum(t['state']=='oriented_diagnostic_view' for t in transforms),
                 fresh_readings=len(rows),rejections=len(reject),prior=summarize(prior['combined']),fresh=summarize(fresh),combined=summarize(combined)))
            assert image_binding(source)==binding
        assert all(sha256(p)==d for p,d in {**pins,**models,**old_protocol['pins']}.items()) and source_pins()==before
        final=dict(status='complete',cases=totals,fresh_OCR_calls=calls,seconds=time.monotonic()-start,mainline_unchanged=True,
                   new_confirmed_connections=0,manual_review='pending',field_accuracy=None,no_deployment=True)
        save(OUT/'report.json',final);save(OUT/'progress.json',final);print(json.dumps(final))
    except BaseException as exc:
        save(OUT/'progress.json',dict(status='failed',error=type(exc).__name__+': '+str(exc),calls_completed=calls,seconds=time.monotonic()-start));raise


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
