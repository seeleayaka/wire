"""Verify every new scope mask, render all, nominate visible attachments conservatively."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core import image_binding, sha256
from run_audit import source_pins
from run_paired_evidence import load_run, lift_verified_context
from run_prompt_contrast import digest, verify
from run_review import save, verified_run
from visible_lead_scope import nominate_view, validate_scope

ROOT=Path(__file__).resolve().parents[2]


def render_all(origin,records,directory):
    directory.mkdir(exist_ok=False)
    with Image.open(origin['image_path']) as opened:rgb=np.asarray(opened.convert('RGB'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    columns=4;cw,ch=430,460
    sheet=Image.new('RGB',(columns*cw,max(1,(len(records)+columns-1)//columns)*ch),'#edf0f3')
    for index,record in enumerate(records):
        geometry=record['geometry']
        with Image.open(record['source_mask_path']) as opened:active=np.asarray(opened.convert('L'))>0
        eligible=(record['score']>=.75 and geometry['state']=='simple_visible_path'
                  and not record['boundary_truncated'])
        color=np.array([30,155,100] if eligible else [215,125,30],float)
        overlay=rgb.astype(float).copy();overlay[active]=overlay[active]*.6+color*.4
        panel=Image.new('RGB',(cw,ch),'white');photo=Image.fromarray(overlay.astype('uint8'))
        photo.thumbnail((410,405));panel.paste(photo,(5,48));draw=ImageDraw.Draw(panel)
        draw.text((6,4),f"{record['record_id']}  score {record['score']:.3f}",font=font,fill='#263342')
        draw.text((6,24),geometry['state']+(' | boundary' if record['boundary_truncated'] else ''),font=font,fill='#805715')
        panel.save(directory/(record['record_id']+'.png'))
        sheet.paste(panel,((index%columns)*cw,(index//columns)*ch))
    sheet.save(directory/'all_masks.png')


def main():
    evidence=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    output=ROOT/'artifacts/mendeley_visible_fan_scope_analysis_20261005'
    if output.exists():raise FileExistsError('preserve prior analysis')
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((evidence/'inference_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete' or inference['protocol_sha256']!=digest(evidence/'protocol.json'):
        raise ValueError('fresh inference not completed or protocol drift')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E mainline or dirty status drift')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    ref_binding=protocol['original_sources']['reference']
    reference_binding={k:ref_binding[k] for k in ['image_sha256','image_size','coordinate_frame']}
    validate_scope(scope,reference_binding)
    registration=json.loads((evidence/'registration/report.json').read_text(encoding='utf-8'))['registration']
    if not registration['alignment_quality']['reliable']:raise ValueError('global registration failed')
    context=json.loads((evidence/'context_protocol.json').read_text(encoding='utf-8'))
    extra=[]
    for case in context['cases']:
        wire=deepcopy(case);wire['fresh_run_directory']=str(Path(case['fresh_run_directory']).with_name('wire'))
        extra.append(wire)
    context['cases'].extend(extra)
    output.mkdir(exist_ok=False)
    save(output/'exact_context_for_all_prompts.json',context)
    cases=[]
    for side in ['reference','inspection']:
        for prompt in ['cable','wire']:
            run=evidence/side/prompt
            origin,records=load_run(run)
            render_all(origin,records,output/(side+'_'+prompt))
            full,lifted=lift_verified_context(origin,records,output/'exact_context_for_all_prompts.json')
            nomination=nominate_view(scope,reference_binding,lifted,
                inspection_to_reference=None if side=='reference' else registration['source_to_reference_homography'])
            verified_run(run,origin['image_path'])
            cases.append({'side':side,'prompt':prompt,'mask_count':len(records),
                'geometry_counts':dict(Counter(r['geometry']['state'] for r in records)),
                'strict_eligible_whole_masks':sum(r['score']>=.75 and r['geometry']['state']=='simple_visible_path'
                    and not r['boundary_truncated'] for r in records),
                'nomination':nomination,'origin':full})
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('mainline drift during analysis')
    summary={'status':'complete','new_mask_instances':sum(c['mask_count'] for c in cases),
        'fresh_image_encoders':inference['fresh_image_encoders'],'fresh_prompt_decoders':inference['new_prompt_decoders'],
        'inference_seconds':inference['seconds'],'cases':cases,
        'reference_review_confirmed':False,'new_confirmed_visible_attachments':0,
        'new_confirmed_electrical_connections':0,'model_observer_count':1,
        'decision':'insufficient_evidence','reason':'no_complete_unique_two_anchor_path_in_reference_or_inspection',
        'GT_read':False,'mainline_and_original_inputs_unchanged':True,'deployed':False,
        'reference_internal_fan_terminal_visible':False,
        'scope_draft_sha256':sha256(scope_path),
        'protocol_sha256':sha256(evidence/'protocol.json'),
        'inference_report_sha256':sha256(evidence/'inference_report.json'),
        'analysis_source_pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('visible_lead_scope.py'),
            Path(__file__).with_name('run_paired_evidence.py'),Path(__file__).with_name('run_review.py'),
            Path(__file__).with_name('core.py')]}}
    save(output/'report.json',summary)
    print(json.dumps({'status':'complete','instances':summary['new_mask_instances'],
        'cases':[{'side':c['side'],'prompt':c['prompt'],'masks':c['mask_count'],
            'eligible':c['strict_eligible_whole_masks'],'nominations':len(c['nomination']['relation_nominations'])} for c in cases],
        'decision':summary['decision']},ensure_ascii=False))


if __name__=='__main__':main()
