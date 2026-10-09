"""Verify all fresh alternative-prompt outputs, preserve physical-wire gates."""
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
from run_review import save
from visible_lead_scope import nominate_view

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_prompt_semantics_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((source/'inference_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(source/'protocol.json'):
        raise ValueError('fresh inference incomplete or protocol drift')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E dirty tree or source drift')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));binding=scope['reference_binding']
    registration_path=ROOT/'artifacts/mendeley_visible_fan_scope_20261005/registration/report.json'
    registration=json.loads(registration_path.read_text(encoding='utf-8'))['registration']
    if not registration['alignment_quality']['reliable']:raise ValueError('global registration failed')
    output=ROOT/'artifacts/mendeley_prompt_semantics_analysis_20261005'
    if output.exists():raise FileExistsError('preserve previous analysis')
    output.mkdir(exist_ok=False);context={'cases':[]}
    for case in protocol['cases']:
        for recipe in protocol['recipes']:
            context['cases'].append({'fresh_run_directory':str(source/case['id']/recipe),
                'crop_box_xyxy':case['crop_box_xyxy'],'crop_binding':case['source'],
                'source_binding':dict(case['original_source'],image_path=case['original_source']['path'])})
    save(output/'exact_context.json',context);results=[]
    for case in protocol['cases']:
        side=case['id'];matrix=None if side=='reference' else registration['source_to_reference_homography']
        for recipe in protocol['recipes']:
            origin,records=load_run(source/side/recipe)
            render_all(origin,records,output/(side+'_'+recipe))
            full,lifted=lift_verified_context(origin,records,output/'exact_context.json')
            strict=nominate_view(scope,binding,lifted,inspection_to_reference=matrix)
            diagnostic=[]
            for record in records:
                with Image.open(record['source_mask_path']) as opened:raw=np.asarray(opened.convert('L'))
                diagnostic.append(inspect_support(raw,record['score'],record['record_id'],scope,binding,
                    tuple(case['crop_box_xyxy'][:2]),matrix))
            results.append({'side':side,'recipe':recipe,'masks':len(records),'strict_nomination':strict,
                'whole_mask_geometry':[{'record_id':r['record_id'],'score':r['score'],
                    'geometry':r['geometry'],'boundary_truncated':r['boundary_truncated']} for r in records],
                'bundle_pixel_support_diagnostics':diagnostic,'origin':full})
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('mainline drift during analysis')
    report={'status':'complete','results':results,'new_mask_instances':sum(r['masks'] for r in results),
        'fresh_encoders':inference['fresh_encoders'],'fresh_decoder_calls':inference['fresh_decoder_calls'],
        'inference_seconds':inference['seconds'],'decision':'insufficient_evidence',
        'strict_accepted_attachment_count':0,'new_confirmed_electrical_connections':0,
        'reference_review_confirmed':False,'model_observer_count':1,'GT_read':False,'deployed':False,
        'thresholds_changed':False,'no_branch_pruning_or_gap_filling':True,
        'pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('bundle_support.py'),
            Path(__file__).with_name('visible_lead_scope.py'),Path(__file__).with_name('core.py'),
            Path(__file__).with_name('run_paired_evidence.py'),source/'protocol.json',source/'inference_report.json',
            scope_path,registration_path]}}
    save(output/'report.json',report)
    print(json.dumps({'status':'complete','seconds':report['inference_seconds'],
        'masks':report['new_mask_instances'],'results':[{'side':r['side'],'recipe':r['recipe'],
            'strict_nominations':len(r['strict_nomination']['relation_nominations']),
            'records':[{'id':d['record_id'],'score':d['score'],'state':d['whole_mask_geometry']['state'],
                'components':d['raw_foreground_component_count'],'tips':d['whole_mask_geometry']['tip_count'],
                'both_anchor_support':d['component_has_both_anchor_support']} for d in r['bundle_pixel_support_diagnostics']]
            } for r in results]},ensure_ascii=False))


if __name__=='__main__':main()
