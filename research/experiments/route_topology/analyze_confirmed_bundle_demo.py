"""Verified new native SAM + fresh local poses + typed socket/bundle review."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from analyze_mendeley_scope import render_all
from core import sha256
from run_audit import source_pins
from run_paired_evidence import load_run,lift_verified_context
from run_prompt_contrast import verify
from run_review import save,verified_run
from visible_lead_scope import nominate_view
from visible_bundle_relation import observe_bundle,compare_bundle

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_confirmed_bundle_demo_20261006'
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((evidence/'inference_report.json').read_text(encoding='utf-8'))
    preparation=json.loads((evidence/'preparation_report.json').read_text(encoding='utf-8'))
    scope=json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(evidence/'protocol.json'):raise ValueError('fresh SAM not complete or drift')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    output=ROOT/'artifacts/mendeley_confirmed_bundle_review_20261006';output.mkdir(exist_ok=False)
    context={'cases':[{'fresh_run_directory':str(evidence/c['id']/recipe),'crop_box_xyxy':c['crop_box_xyxy'],
        'crop_binding':c['source'],'source_binding':dict(c['original_source'],image_path=c['original_source']['path'])}
        for c in protocol['cases'] for recipe in protocol['recipes']]}
    save(output/'exact_context.json',context)
    observations=[];all_mask_count=0;strict_counts={};native_records={}
    for row in preparation['cases']:
        case=row.get('crop_context');binding={k:row['original_source'][k] for k in ['image_sha256','image_size','coordinate_frame']}
        if sha256(row['original_source']['path'])!=binding['image_sha256']:raise ValueError('original image changed')
        masks=[];verified=False
        if row['sam_inference_requested']:
            with Image.open(row['original_source']['path']) as im:original=np.asarray(im.convert('RGB').crop(case['crop_box_xyxy']))
            with Image.open(case['source']['path']) as im:crop=np.asarray(im.convert('RGB'))
            if not np.array_equal(original,crop):raise ValueError('SAM crop not exact original pixels')
            for recipe in protocol['recipes']:
                run=evidence/row['id']/recipe;origin,records=load_run(run)
                render_all(origin,records,output/(row['id']+'_'+recipe));all_mask_count+=len(records)
                verified_run(run,origin['image_path'])
                _,lifted=lift_verified_context(origin,records,output/'exact_context.json')
                strict=nominate_view(scope,scope['reference_binding'],lifted,
                    inspection_to_reference=np.array(row['registration']['source_to_reference_homography']))
                strict_counts[row['id']+'_'+recipe]=len(strict['relation_nominations'])
                if recipe=='cable_plus_reference_anatomy_box':
                    native_records[row['id']]=records
                    for r in records:
                        with Image.open(r['source_mask_path']) as im:raw=np.asarray(im.convert('L'))
                        masks.append({'raw':raw,'record_id':r['record_id'],'score':r['score'],'recipe':recipe})
            verified=True
        observation=observe_bundle(scope,binding,row['anchors'],masks,row['phenotype'],
            translation=tuple(case['crop_box_xyxy'][:2]) if case is not None else (0,0),sam_inventory_verified=verified)
        observations.append({'id':row['id'],'observation':observation})
    reference=observations[0]['observation']
    results=[{'id':r['id'],'comparison':compare_bundle(scope,reference,r['observation']),
        'observation':r['observation']} for r in observations[1:]]
    names={'same_visible_bundle_attachment_supported':'线束可见接法与参考一致',
        'visible_socket_attachment_change_supported':'插座可见插接发生变化','insufficient_evidence':'证据不足，不判断'}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    sheet=Image.new('RGB',(1440,700),'#eef1f4');draw=ImageDraw.Draw(sheet)
    draw.text((14,10),'已确认正常参考｜三种实际原图流程结果（线束级可见关系，不是电气导通）',font=font,fill='#293746')
    for i,result in enumerate(results):
        row=next(r for r in preparation['cases'] if r['id']==result['id']);decision=result['comparison']['decision'];case=row['crop_context']
        with Image.open(case['source']['path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        eligible=result['observation']['eligible_native_components']
        if len(eligible)==1:
            r=next(r for r in native_records[row['id']] if r['record_id']==eligible[0]['record_id'])
            with Image.open(r['source_mask_path']) as im:active=np.asarray(im.convert('L'))>0
            overlay=rgb.astype(float);overlay[active]=overlay[active]*.72+np.array([50,163,123])*.28;rgb=overlay.astype(np.uint8)
        x=i*480;draw.rectangle((x+8,70,x+472,690),fill='white')
        color='#236b60' if decision=='same_visible_bundle_attachment_supported' else '#b84e34' if decision=='visible_socket_attachment_change_supported' else '#8a7130'
        draw.text((x+16,85),row['id']+'  '+names[decision],font=font,fill=color)
        photo=Image.fromarray(rgb);photo.thumbnail((440,440));sheet.paste(photo,(x+18,135))
        with Image.open(row['socket_patch_path']) as im:patch=im.convert('RGB').resize((400,100),Image.Resampling.NEAREST)
        sheet.paste(patch,(x+38,561))
        draw.text((x+16,667),'不判内部接点／逐芯接法／电气导通',font=small,fill='#775b30')
    sheet.save(output/'decision_matrix.png')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift during report')
    report={'status':'complete','reference_observation':reference,'cases':results,'reference_review_confirmed':True,
        'fresh_original_images':4,'fresh_inspection_global_and_local_registration':3,
        'fresh_SAM_encoders':inference['fresh_encoders'],'fresh_SAM_decoders':inference['fresh_decoders'],
        'SAM_output_instances':all_mask_count,'inference_seconds':inference['seconds'],
        'old_single_wire_geometry_nomination_counts':strict_counts,'old_single_wire_gate_unchanged':True,
        'new_scope':'reference_once_multiwire_bundle_visible_attachment_review',
        'single_wire_connections_confirmed':0,'new_confirmed_electrical_connections':0,
        'electrical_disconnections_confirmed':0,'deployed_to_E_mainline':False,
        'posthoc_selected_demonstration_not_blind_validation_or_field_accuracy':True,
        'mask_pixels_modified_for_evidence':False,'image_overlay_is_render_only':True,
        'pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('visible_bundle_relation.py'),
            Path(__file__).with_name('visible_lead_scope.py'),Path(protocol['confirmed_scope_path']),
            evidence/'protocol.json',evidence/'preparation_report.json',evidence/'inference_report.json']}}
    save(output/'report.json',report)
    print(json.dumps({'status':'complete','fresh_SAM_encoders':report['fresh_SAM_encoders'],
        'cases':{r['id']:r['comparison']['decision'] for r in results},'single_wire_confirmed':0,'electrical_connections_confirmed':0},ensure_ascii=False))


if __name__=='__main__':main()
