"""Negative-control native mask audit, no invented path or anchor confirmation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from analyze_mendeley_scope import render_all
from bundle_support import inspect_support
from core import sha256
from run_audit import source_pins
from run_paired_evidence import load_run,lift_verified_context
from run_prompt_contrast import verify
from run_review import save,verified_run
from visible_lead_scope import nominate_view

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_exposed_socket_sam_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((source/'inference_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(source/'protocol.json'):raise ValueError('SAM incomplete or protocol drift')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));binding=scope['reference_binding']
    preview_path=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_20261005/report.json'
    preview=json.loads(preview_path.read_text(encoding='utf-8'))
    row=next(r for r in preview['cases'] if r['id']=='case_06');case=protocol['cases'][0]
    if case['original_source']['image_sha256']!=row['input_sha256']:raise ValueError('original image evidence mismatch')
    with Image.open(case['original_source']['path']) as im:fresh=np.asarray(im.convert('RGB').crop(case['crop_box_xyxy']))
    with Image.open(case['source']['path']) as im:saved=np.asarray(im.convert('RGB'))
    if not np.array_equal(fresh,saved):raise ValueError('SAM input not exact original RGB crop')
    h=np.array(row['registration']['source_to_reference_homography'])
    output=ROOT/'artifacts/mendeley_exposed_socket_sam_analysis_20261005';output.mkdir(exist_ok=False)
    save(output/'exact_context.json',{'cases':[{'fresh_run_directory':str(source/'case_06'/recipe),
        'crop_box_xyxy':case['crop_box_xyxy'],'crop_binding':case['source'],
        'source_binding':dict(case['original_source'],image_path=case['original_source']['path'])} for recipe in protocol['recipes']]})
    analyses=[]
    for recipe in protocol['recipes']:
        run=source/'case_06'/recipe;origin,records=load_run(run)
        render_all(origin,records,output/recipe)
        _,lifted=lift_verified_context(origin,records,output/'exact_context.json')
        strict=nominate_view(scope,binding,lifted,inspection_to_reference=h)
        support=[]
        for record in records:
            with Image.open(record['source_mask_path']) as im:raw=np.asarray(im.convert('L'))>0
            support.append(inspect_support(raw,record['score'],record['record_id'],scope,binding,
                translation=tuple(case['crop_box_xyxy'][:2]),matrix=h))
        verified_run(run,origin['image_path'])
        analyses.append({'recipe':recipe,'mask_count':len(records),'strict_geometry_diagnostics':strict,
            'raw_component_support':support,'fan_anchor_localization_failed':True,
            'relation_nominations_accepted':0,'decision':'insufficient_evidence'})
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift during analysis')
    suspicious=[{'recipe':a['recipe'],'record_id':r['record_id'],'score':r['score']} for a in analyses for r in a['raw_component_support'] if r['score']>=.75 and r['component_has_both_anchor_support']]
    result={'status':'complete','cases':analyses,'fresh_encoders':inference['fresh_encoders'],
        'fresh_decoders':inference['fresh_decoders'],'inference_seconds':inference['seconds'],
        'mask_count':sum(a['mask_count'] for a in analyses),'high_score_masks_touching_both_draft_anchors':suspicious,
        'socket_phenotype':'socket_contacts_exposed','fan_anchor_localization_qualified':False,
        'reference_review_confirmed':False,'new_confirmed_connections':0,'confirmed_disconnections':0,
        'decision':'insufficient_evidence','model_observer_count':1,'deployed':False,
        'sample_selected_posthoc_for_negative_control_not_accuracy':True,
        'mask_pixel_modification_or_branch_pruning':False,
        'pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('bundle_support.py'),
            Path(__file__).with_name('run_paired_evidence.py'),Path(__file__).with_name('visible_lead_scope.py'),
            scope_path,preview_path,source/'protocol.json',source/'inference_report.json']}}
    save(output/'report.json',result);print(json.dumps({k:result[k] for k in ['status','mask_count','high_score_masks_touching_both_draft_anchors','new_confirmed_connections']}))


if __name__=='__main__':main()
