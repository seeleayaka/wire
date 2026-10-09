"""Audit fresh geometry prompts without accepting branched bundle masks as wires."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_mendeley_scope import render_all
from core import sha256
from run_audit import source_pins
from run_paired_evidence import load_run, lift_verified_context
from run_prompt_contrast import verify
from run_review import save, verified_run
from visible_lead_scope import nominate_view, validate_scope

ROOT=Path(__file__).resolve().parents[2]


def tip_diagnostic(scope,records,matrix):
    """Unaccepted two-tip diagnostics preserve branches and every instance."""
    observations=[]
    for record in records:
        geometry=record['geometry']; tips=np.asarray(geometry['tips_xy'],float)
        if geometry['component_count']!=1 or tips.shape!=(2,2):continue
        q=np.column_stack((tips,np.ones(2)))@matrix.T
        if np.any(np.abs(q[:,2])<1e-9) or np.prod(q[:,2])<=0:continue
        mapped=q[:,:2]/q[:,2:]
        hits=[[a['id'] for a in scope['anchors'] if
            a['bbox_xyxy'][0]<=x<=a['bbox_xyxy'][2] and
            a['bbox_xyxy'][1]<=y<=a['bbox_xyxy'][3]] for x,y in mapped]
        observations.append({'record_id':record['record_id'],'score':record['score'],
            'original_geometry_state':geometry['state'],'branch_pixel_count':geometry['branch_pixel_count'],
            'boundary_truncated':record['boundary_truncated'],'reference_frame_tips_xy':mapped.tolist(),
            'anchor_hits':hits,'two_unique_anchor_hits':all(len(v)==1 for v in hits) and hits[0]!=hits[1],
            'accepted':False,'semantics_verified':False,'physical_continuity_verified':False})
    return observations


def main():
    evidence=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    output=ROOT/'artifacts/mendeley_reference_geometry_analysis_20261005'
    if output.exists():raise FileExistsError('preserve prior analysis')
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((evidence/'inference_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete' or inference['protocol_sha256']!=sha256(evidence/'protocol.json'):
        raise ValueError('fresh inference incomplete or protocol drift')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('mainline drift')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));binding=scope['reference_binding']
    validate_scope(scope,binding)
    registration_path=ROOT/'artifacts/mendeley_visible_fan_scope_20261005/registration/report.json'
    registration=json.loads(registration_path.read_text(encoding='utf-8'))['registration']
    if not registration['alignment_quality']['reliable']:raise ValueError('global registration rejected')
    h=np.asarray(registration['source_to_reference_homography'],float)
    output.mkdir(exist_ok=False)
    context={'cases':[]}
    for case in protocol['cases']:
        for recipe in protocol['recipes']:
            context['cases'].append({'fresh_run_directory':str(evidence/case['id']/recipe),
                'crop_box_xyxy':case['crop_box_xyxy'],
                'crop_binding':case['source'],
                'source_binding':dict(case['original_source'],image_path=case['original_source']['path'])})
    save(output/'exact_context.json',context)
    cases=[];panels=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for case in protocol['cases']:
        side=case['id'];matrix=np.eye(3) if side=='reference' else h
        for recipe in protocol['recipes']:
            run=evidence/side/recipe;origin,records=load_run(run)
            render_all(origin,records,output/(side+'_'+recipe))
            full,lifted=lift_verified_context(origin,records,output/'exact_context.json')
            strict=nominate_view(scope,binding,lifted,inspection_to_reference=matrix)
            control=None
            if recipe=='cable':
                _,previous=load_run(ROOT/'artifacts/mendeley_visible_fan_scope_20261005'/side/'cable')
                signature=lambda rs:[(r['mask_array_sha256'],r['score']) for r in rs]
                control=signature(records)==signature(previous)
                if not control:raise ValueError('fresh text control differs from previous exact-pixel control')
            diagnostics=tip_diagnostic(scope,lifted,matrix)
            for diagnostic in diagnostics:
                if not diagnostic['two_unique_anchor_hits'] or diagnostic['score']<.75:continue
                record=next(r for r in records if r['record_id']==diagnostic['record_id'])
                with Image.open(origin['image_path']) as opened:rgb=np.asarray(opened.convert('RGB')).copy()
                with Image.open(record['source_mask_path']) as opened:active=np.asarray(opened.convert('L'))>0
                overlay=rgb.astype(float);overlay[active]=overlay[active]*.65+np.array([245,164,60])*.35
                panel=Image.new('RGB',(450,475),'white');photo=Image.fromarray(overlay.astype('uint8'))
                photo.thumbnail((430,410));panel.paste(photo,(10,60));draw=ImageDraw.Draw(panel)
                draw.text((10,5),side+' | '+record['record_id']+' | '+str(record['score']),font=font,fill='#223343')
                draw.text((10,28),'整束分割候選；非單線/電氣連通認定',font=font,fill='#986010')
                for x,y in record['geometry']['tips_xy']:draw.ellipse((10+x-4,60+y-4,10+x+4,60+y+4),outline='#bf2020',width=2)
                panels.append(panel)
            verified_run(run,origin['image_path'])
            cases.append({'side':side,'recipe':recipe,'mask_count':len(records),
                'geometry_counts':dict(Counter(r['geometry']['state'] for r in records)),
                'strict_nomination':strict,'two_tip_diagnostics':diagnostics,
                'text_control_exact_masks_and_scores_reproduced':control,'origin':full})
    if panels:
        sheet=Image.new('RGB',(450*len(panels),475),'#edf0f3')
        for i,panel in enumerate(panels):sheet.paste(panel,(450*i,0))
        sheet.save(output/'two_anchor_bundle_candidates.png')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('mainline drift during audit')
    report={'status':'complete','cases':cases,'new_mask_instances':sum(c['mask_count'] for c in cases),
        'fresh_encoders':inference['fresh_encoders'],'fresh_decoders':inference['fresh_decoders'],
        'inference_seconds':inference['seconds'],'strict_accepted_attachment_count':0,
        'new_confirmed_electrical_connections':0,'decision':'insufficient_evidence',
        'reference_review_confirmed':False,'local_anchor_identity_not_fully_verified':True,
        'model_observer_count':1,'deployed':False,'GT_read':False,
        'branch_pruning':False,'fragment_bridging':False,'thresholds_changed':False,
        'two_tip_diagnostics_are_posthoc_not_validation':True,
        'protocol_sha256':sha256(evidence/'protocol.json'),
        'inference_report_sha256':sha256(evidence/'inference_report.json'),
        'scope_draft_sha256':sha256(scope_path),
        'analysis_source_pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('analyze_mendeley_scope.py'),
            Path(__file__).with_name('visible_lead_scope.py'),Path(__file__).with_name('run_paired_evidence.py'),
            Path(__file__).with_name('core.py')]}}
    save(output/'report.json',report)
    print(json.dumps({'status':'complete','instances':report['new_mask_instances'],
        'cases':[{'side':c['side'],'recipe':c['recipe'],'masks':c['mask_count'],
            'strict_nominations':len(c['strict_nomination']['relation_nominations']),
            'two_tip_diagnostics':c['two_tip_diagnostics']} for c in cases]},ensure_ascii=False))


if __name__=='__main__':main()
