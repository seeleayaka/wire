"""Fixed generic mask-context OCR trial. Old bound SAM; NEW local OCR calls.

No SAM retraining/inference, terminal declarations, nearest assignment, token
repair, GT reads, per-image tuning or deployment. Baseline raw evidence kept.
"""
from datetime import datetime,timezone
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from core import image_binding,sha256
from identity_ocr import inverse_rotation
from label_attachment import nominate,mask_context_crop,polygon_pixels
from run_paired_evidence import load_run,lift_verified_context
from run_audit import source_pins
from run_identity_ocr import DEPENDENCIES
from run_review import save,read

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/route_mask_label_ocr_20261005'
CONTEXT=ROOT/'artifacts/sam_crop_coverage_20261001/frozen_protocol.json'


def prepare_cases():
    cases=[]
    for name in ['cabinet_1','cabinet_2']:
        run=ROOT/'artifacts/sam_crop_coverage_20261001'/f'{name}_crop_run'
        origin,records=load_run(run)
        full,lifted=lift_verified_context(origin,records,CONTEXT)
        width,height=full['frame_binding']['image_size']
        x0,y0,x1,y1=full['verified_crop_box_xyxy']
        masks=[]
        for record in records:
            with Image.open(record['source_mask_path']) as opened:
                pixels=np.asarray(opened.convert('L'))>0
            global_pixels=np.zeros((height,width),bool)
            global_pixels[y0:y1,x0:x1]=pixels
            eligible=(record['score']>=.75 and not record['boundary_truncated'] and
                      record['geometry']['state']=='simple_visible_path')
            masks.append({'id':record['record_id'],'pixels':global_pixels,
                          'topology_geometry_eligible':eligible,
                          'crop_boundary_guards_preserved':True})
        baseline_path=ROOT/'artifacts/route_identity_ocr_tiles_20261005'/name/'report.json'
        baseline=read(baseline_path)
        if baseline['image_binding']!=full['frame_binding']:
            raise ValueError('baseline OCR belongs to different source frame')
        cases.append({'id':name,'origin':origin,'full':full,'records':records,
                      'masks':masks,'baseline_path':baseline_path,'baseline':baseline})
    return cases


def valid_row(row,size):
    polygon_pixels(row['polygon_source_xy'],size)
    p=np.asarray(row['polygon_source_xy'],float)
    box=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
    if not np.allclose(box,row['bbox_xyxy'],rtol=0,atol=1e-8):
        raise ValueError('polygon/bbox mismatch')


def summarize(result):
    return {'spatial_groups':len(result['groups']),
            'unique_pixel_supported_groups':sum(g['nominated_mask_id'] is not None for g in result['groups']),
            'high_score_consistent_attached_groups':sum(g['high_score_consistent_text_on_mask'] for g in result['groups']),
            'high_score_ascii_attached_groups':sum(g['high_score_consistent_text_on_mask'] and g['text_group']['simple_ascii_identity_candidate'] for g in result['groups']),
            'membership_states':dict(Counter(a['state'] for a in result['reading_memberships'].values())),
            'repeated_same_text_masks':result['same_text_different_masks'],
            'confirmed_wire_or_port_identities':0,'confirmed_connections':0}


def main():
    if OUT.exists():
        raise FileExistsError('preserve old outputs')
    cases=prepare_cases()
    models={str(p.resolve()):sha256(p) for p in sorted((DEPENDENCIES/'rapidocr_onnxruntime/models').glob('*.onnx'))}
    if len(models)!=3:
        raise ValueError('requires the same 3 pinned offline OCR weights')
    code=[Path(__file__),*[Path(__file__).with_name(n) for n in
          ['label_attachment.py','identity_ocr.py','run_paired_evidence.py','core.py','run_review.py','run_audit.py','run_identity_ocr.py']]]
    pins={str(p.resolve()):sha256(p) for p in code+[CONTEXT]}
    for case in cases:
        pins[str(case['baseline_path'].resolve())]=sha256(case['baseline_path'])
        for item in case['origin']['mask_files']:
            pins[item['path']]=item['sha256']
        pins[str(Path(case['origin']['directory'])/'run_manifest.json')]=case['origin']['manifest_sha256']
    before=source_pins()
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
         'sources':{c['id']:{'image_path':c['full']['image_path'],**c['full']['frame_binding'],
                    'bound_SAM_run':c['origin']['directory'],'crop_box':c['full']['verified_crop_box_xyxy'],
                    'instance_count':len(c['masks'])} for c in cases},
         'models':models,'pins':pins,'mainline_pins':before,
         'mask_context_padding':'max(4, ceil(0.5 * shorter native bbox side))',
         'text_crop_boundary_margin_source_pixels':2,'scale':2,'quarter_turns':[0,1,2,3],
         'membership':'any other nonidentical mask intersection rejects; unique mask must cover >=0.5 of polygon raster',
         'spatial_grouping':'unchanged pairwise IoU0.5, high score0.9 and no conflicting text variants',
         'no_nearest_assignment':True,'same_OCR_weights_count_as_one_observer':True,
         'same_SAM_weights_count_as_one_observer':True,'old_SAM_reused_verified':True,
         'fresh_SAM_this_experiment':False,'fresh_OCR_this_experiment':True,
         'no_GT_reads_or_token_repair':True,'no_per_image_ROI_or_parameter_sweep':True,
         'no_confirmed_identity_or_expected_edges':True,'E_deployment':False})
    start=time.perf_counter();calls=0;summary=[]
    try:
        sys.path.insert(0,str(DEPENDENCIES))
        from rapidocr_onnxruntime import RapidOCR
        engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=2)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        for case in cases:
            target=OUT/case['id'];target.mkdir()
            source=case['full']['image_path'];binding=case['full']['frame_binding']
            if image_binding(source)!=binding:
                raise ValueError('source drift before OCR')
            with Image.open(source) as opened:
                original=opened.convert('RGB')
            rows=[];rejected=[];crops=[]
            for mask in case['masks']:
                box=mask_context_crop(mask['pixels'],original.size)
                if box is None:
                    crops.append({'mask_id':mask['id'],'state':'empty_mask_no_crop'})
                    continue
                crop=original.crop(box)
                doubled=np.asarray(crop.resize((crop.width*2,crop.height*2),Image.Resampling.LANCZOS))
                crops.append({'mask_id':mask['id'],'source_crop_xyxy':box,
                              'topology_geometry_eligible':mask['topology_geometry_eligible']})
                for rotation in range(4):
                    save(OUT/'progress.json',{'status':'new_OCR','case':case['id'],'mask':mask['id'],
                         'rotation':rotation,'calls_completed':calls,'cases_completed':len(summary),
                         'elapsed_seconds':time.perf_counter()-start})
                    result,elapsed=engine(np.ascontiguousarray(np.rot90(doubled,rotation)[:,:,::-1]))
                    calls+=1
                    view=f'{mask["id"]}_v{rotation}'
                    save(target/f'{view}_raw.json',{'rows':result,'elapsed':elapsed,
                                                   'source_crop_xyxy':box})
                    for index,(polygon,text,score) in enumerate(result or []):
                        p=np.asarray(inverse_rotation(polygon,crop.width*2,crop.height*2,rotation,2))
                        local=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
                        rid=f'new_{view}_{index:03d}'
                        if local[0]<=2 or local[1]<=2 or local[2]>=crop.width-2 or local[3]>=crop.height-2:
                            rejected.append({'record_id':rid,'reason':'text_touches_crop_boundary','text':text,'score':float(score)})
                            continue
                        p+=np.array(box[:2])
                        row={'record_id':rid,'text':text,'score':float(score),
                             'polygon_source_xy':p.tolist(),
                             'bbox_xyxy':[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())],
                             'view_quarter_turns':rotation,'source_crop_xyxy':box,
                             'crop_proposer_mask_id':mask['id'],'evidence_origin':'fresh_mask_context_OCR'}
                        try:
                            valid_row(row,original.size)
                        except ValueError as exc:
                            rejected.append({'record_id':rid,'reason':str(exc),'text':text,'score':float(score)})
                            continue
                        rows.append(row)
            baseline=[];baseline_rejected=[]
            for old in case['baseline']['records']:
                row=dict(old,record_id='base_'+old['record_id'],evidence_origin='previous_fixed_tiles_OCR')
                try:
                    valid_row(row,original.size)
                except ValueError as exc:
                    baseline_rejected.append({'record_id':row['record_id'],'reason':str(exc)})
                    continue
                baseline.append(row)
            old_result=nominate(baseline,case['masks'],original.size)
            new_result=nominate(rows,case['masks'],original.size)
            combined=nominate(baseline+rows,case['masks'],original.size)
            detailed={'image_binding':binding,'image_path':source,'baseline_raw_readings_preserved':case['baseline']['records'],
                      'baseline_analysis_rejections':baseline_rejected,'fresh_readings':rows,'new_rejections':rejected,
                      'mask_context_crops':crops,'baseline':old_result,'fresh':new_result,'combined':combined,
                      'all_original_source_and_crop_boundary_guards_preserved':True,
                      'fresh_SAM':False,'fresh_OCR':True,'topology_decision':'insufficient_evidence',
                      'physical_connection_accuracy':None,'deployed':False}
            save(target/'report.json',detailed)
            canvas=original.resize((original.width*2,original.height*2),Image.Resampling.LANCZOS)
            draw=ImageDraw.Draw(canvas)
            for nomination in new_result['groups']:
                group=nomination['text_group'];box=[v*2 for v in group['bbox_xyxy']]
                color='#007e70' if nomination['high_score_consistent_text_on_mask'] else '#b76b00'
                draw.rectangle(box,outline=color,width=2)
                label=f"{group['best_text']} → {nomination['nominated_mask_id'] or '?'}"
                draw.text((box[0],max(0,box[1]-18)),label,font=font,fill=color,stroke_width=1,stroke_fill='white')
            canvas.save(target/'fresh_text_on_mask.png')
            summary.append({'case':case['id'],'source_masks':len(case['masks']),
                 'new_retained_readings':len(rows),'new_boundary_or_geometry_rejections':len(rejected),
                 'baseline':summarize(old_result),'fresh':summarize(new_result),'combined':summarize(combined)})
            if image_binding(source)!=binding:
                raise ValueError('source drift after OCR')
        if any(sha256(p)!=h for p,h in {**models,**pins}.items()) or source_pins()!=before:
            raise ValueError('source/model/experiment/mainline drift')
        final={'status':'complete','seconds':time.perf_counter()-start,'fresh_local_OCR_calls':calls,
               'cases':summary,'source_model_code_and_mainline_unchanged':True,
               'new_confirmed_connections':0,'field_accuracy':None,'manual_review':'pending',
               'E_deployed':False,'protocol_sha256':sha256(OUT/'protocol.json')}
        save(OUT/'report.json',final);save(OUT/'progress.json',final)
        print(json.dumps(final,ensure_ascii=False))
    except BaseException as exc:
        save(OUT/'progress.json',{'status':'failed','error':str(exc),'calls_completed':calls,'completed_cases':len(summary)})
        (OUT/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
