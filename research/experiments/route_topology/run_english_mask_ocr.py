"""One fixed official English/numeric recognizer contrast, local pixels only."""
import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_mask_label_ocr import prepare_cases,valid_row,summarize,OUT as BASELINE
from label_attachment import nominate,mask_context_crop
from identity_ocr import inverse_rotation
from run_identity_ocr import DEPENDENCIES
from core import sha256,image_binding
from run_audit import source_pins
from run_review import read,save

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/route_english_mask_ocr_20261005'
MODEL=ROOT/'artifacts/route_english_ocr_dependencies_20261005/python_tls/en_PP-OCRv5_rec_mobile.onnx'
MODEL_SHA='c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8'
PLAN=ROOT/'artifacts/route_english_ocr_preregistration_20261005/PLAN.md'


def main():
    if OUT.exists():raise FileExistsError('preserve English recognizer contrast')
    old_protocol=read(BASELINE/'protocol.json')
    if sha256(MODEL)!=MODEL_SHA or read(BASELINE/'report.json')['status']!='complete':raise ValueError('model/baseline prerequisite mismatch')
    if any(sha256(p)!=v for p,v in {**old_protocol['pins'],**old_protocol['models']}.items()):raise ValueError('historical source or inference drift')
    cases=prepare_cases();before=source_pins()
    sdk=DEPENDENCIES/'rapidocr_onnxruntime'
    code=[Path(__file__),PLAN,MODEL,MODEL.with_name('report.json'),MODEL.with_name('official_default_models.yaml'),BASELINE/'protocol.json',BASELINE/'report.json']
    code += [Path(__file__).with_name(n+'.py') for n in ('run_mask_label_ocr','label_attachment','identity_ocr','run_paired_evidence','run_audit','run_review','core')]
    code += list(sdk.rglob('*.py'))+[sdk/'config.yaml']
    pins={str(p):sha256(p) for p in code}
    for case in cases:
        p=BASELINE/case['id']/'report.json';pins[str(p)]=sha256(p)
    OUT.mkdir();sys.path.insert(0,str(DEPENDENCIES))
    from rapidocr_onnxruntime import RapidOCR
    engine=RapidOCR(rec_model_path=str(MODEL),intra_op_num_threads=2,inter_op_num_threads=2)
    if Path(engine.text_rec.session.session._model_path).resolve()!=MODEL.resolve():raise ValueError('actual recognizer binding mismatch')
    tensor=np.zeros((1,3,48,320),np.float32);blank=engine.text_rec.session(tensor)[0]
    if blank.ndim!=3 or blank.shape[2]!=len(engine.text_rec.postprocess_op.character) or not np.isfinite(blank).all():raise ValueError('recognizer decoder mismatch')
    save(OUT/'protocol.json',dict(pins=pins,old_model_pins=old_protocol['models'],mainline_pins=before,recognizer_sha256=MODEL_SHA,
         actual_adapter='RapidOCR1.4.4 local ONNX, replacement recognition only',decoder_classes=blank.shape[2],
         shared_detection_and_classifier=True,logical_OCR_observer_count=1,no_GT_or_token_repair=True,
         fresh_OCR=True,fresh_SAM=False,quarters=[0,1,2,3],scale=2,crop_margin=2,no_deployment=True))
    start=time.monotonic();calls=0;summary=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    try:
        for case in cases:
            folder=OUT/case['id'];folder.mkdir();source=case['full']['image_path'];binding=case['full']['frame_binding']
            if image_binding(source)!=binding:raise ValueError('source pixel drift')
            with Image.open(source) as opened:original=opened.convert('RGB')
            prior=read(BASELINE/case['id']/'report.json');rows=[];rejected=[]
            invalid_old={r['record_id'] for r in prior['baseline_analysis_rejections']}
            old=[dict(r,record_id='base_'+r['record_id']) for r in prior['baseline_raw_readings_preserved'] if 'base_'+r['record_id'] not in invalid_old]+prior['fresh_readings']
            for mask in case['masks']:
                box=mask_context_crop(mask['pixels'],original.size)
                if box is None:continue
                crop=original.crop(box);doubled=np.asarray(crop.resize((2*crop.width,2*crop.height),Image.Resampling.LANCZOS))
                for turn in range(4):
                    save(OUT/'progress.json',dict(status='running',case=case['id'],mask=mask['id'],rotation=turn,calls_completed=calls,seconds=time.monotonic()-start))
                    result,elapsed=engine(np.ascontiguousarray(np.rot90(doubled,turn)[:,:,::-1]));calls+=1
                    view=mask['id']+f'_v{turn}'
                    save(folder/(view+'_raw.json'),dict(rows=result,elapsed=elapsed,source_crop_xyxy=box,recognizer_sha256=MODEL_SHA))
                    for index,(polygon,text,score) in enumerate(result or []):
                        rid='eng5_'+view+f'_{index:03d}'
                        p=np.array(inverse_rotation(polygon,2*crop.width,2*crop.height,turn,2))
                        local=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
                        if local[0]<=2 or local[1]<=2 or local[2]>=crop.width-2 or local[3]>=crop.height-2:
                            rejected.append(dict(record_id=rid,reason='text_touches_crop_boundary',text=text,score=float(score)));continue
                        p+=np.array(box[:2]);bbox=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
                        row=dict(record_id=rid,text=text,score=float(score),polygon_source_xy=p.tolist(),bbox_xyxy=bbox,
                                 view_quarter_turns=turn,crop_proposer_mask_id=mask['id'],source_crop_xyxy=box,
                                 recognizer_sha256=MODEL_SHA,evidence_origin='fresh_English_PP_OCRv5_mask_context')
                        try:valid_row(row,original.size)
                        except ValueError as exc:
                            rejected.append(dict(record_id=rid,reason=str(exc),text=text,score=float(score)));continue
                        rows.append(row)
            fresh=nominate(rows,case['masks'],original.size);combined=nominate(old+rows,case['masks'],original.size)
            save(folder/'report.json',dict(image_path=source,image_binding=binding,old_raw_readings_preserved=old,
                fresh_readings=rows,rejections=rejected,fresh=fresh,combined=combined,prior_summary=summarize(prior['combined']),
                logical_OCR_observer_count=1,recognizers_correlated_not_independent_truth=True,recognizer_sha256=MODEL_SHA,
                confirmed_wire_or_port_identities=0,new_confirmed_connections=0,no_deployment=True))
            canvas=original.resize((original.width*2,original.height*2),Image.Resampling.LANCZOS);draw=ImageDraw.Draw(canvas)
            for g in fresh['groups']:
                text=g['text_group'];bbox=[v*2 for v in text['bbox_xyxy']];color='#087b70' if g['high_score_consistent_text_on_mask'] else '#b76b00'
                draw.rectangle(bbox,outline=color,width=2);draw.text((bbox[0],max(0,bbox[1]-18)),f"{text['best_text']} => {g['nominated_mask_id'] or '?'}",font=font,fill=color,stroke_width=1,stroke_fill='white')
            canvas.save(folder/'fresh_english_text.png')
            summary.append(dict(case=case['id'],source_masks=len(case['masks']),new_readings=len(rows),rejections=len(rejected),
                                prior=summarize(prior['combined']),fresh=summarize(fresh),combined=summarize(combined)))
            if image_binding(source)!=binding:raise ValueError('source drift after inference')
        if calls!=408:raise ValueError('not all mask views processed')
        if any(sha256(p)!=d for p,d in {**pins,**old_protocol['pins'],**old_protocol['models']}.items()) or source_pins()!=before:
            raise ValueError('source/model/code/mainline drift')
        final=dict(status='complete',fresh_OCR_calls=calls,seconds=time.monotonic()-start,cases=summary,
                   mainline_unchanged=True,new_confirmed_connections=0,manual_review='pending',field_accuracy=None,no_deployment=True)
        save(OUT/'report.json',final);save(OUT/'progress.json',final);print(json.dumps(final))
    except BaseException as exc:
        save(OUT/'progress.json',dict(status='failed',error=type(exc).__name__+': '+str(exc),calls_completed=calls,seconds=time.monotonic()-start));raise


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
