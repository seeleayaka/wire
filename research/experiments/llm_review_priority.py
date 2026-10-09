"""Model judgment plus deterministic review-priority gate, never a fault verdict."""
import copy,json,hashlib,math
from pathlib import Path
from urllib.request import Request
import numpy as np
from PIL import Image
import llm_recheck_planner as base

JUDGMENTS={'visible_change','likely_artifact','insufficient_evidence'}
LEVELS={'low':1,'normal':2,'high':3}
LABELS={1:'低优先提示',2:'普通待复核',3:'优先复核'}
SAFE_REASONS={
    'Unexpected fields','Invalid regions','Invalid region','Invalid judgment or priority',
    'Unknown evidence must keep priority','Unexpected plan fields',
    'Missing, duplicate or invented candidate IDs','Incomplete region plan','Unexpected region fields',
    'Invented region','Unknown or duplicate review action/hypothesis','Unsupported confidence',
    'Invalid review text','Duplicate region plan','Invalid summary',
    'DeepSeek model did not return the required JSON object','DeepSeek response JSON must be an object',
    'DeepSeek response has no text content','Disabled',
}

def binding(ref,ins,candidates):
    return {'reference_mask_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),
            'inspection_mask_sha256':hashlib.sha256(ins.read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256(json.dumps(candidates,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}

def payload(ref,ins,candidates,settings):
    request,ids=base.payload(ref,ins,candidates,settings)
    schema={'review_order':ids,'regions':[{'candidate_id':ids[0],
        'observation_zh':'基于可见掩膜的理由','hypotheses':['insufficient_evidence'],
        'requested_checks':['inspect_raw_region'],'question_zh':'请核查局部原图？','confidence':'low',
        'judgment':'insufficient_evidence','suggested_priority':'normal'}],'summary_zh':'复核风险摘要'}
    # Use the existing minimal candidate serializer, never the complete local report.
    _,_,minimal,_=base.backend._mask_inputs(ref,ins,candidates,settings)
    request['messages'][1]['content'][0]['text']=(
        'Return ONLY this JSON schema: '+json.dumps(schema,ensure_ascii=False)+'. '
        'Include all candidates exactly once in regions and review_order. '
        'Judge visible review risk only: judgment is visible_change, likely_artifact, or insufficient_evidence; '
        'suggested_priority is low, normal, or high. Describe concrete visible evidence in observation_zh. '
        'visible_change does NOT mean wiring fault; likely_artifact does NOT mean wiring correct. '
        'If evidence is insufficient, use insufficient_evidence and normal. '
        'No electrical continuity, terminal assignment, cable identity, automatic actions or box changes. '
        'confidence is low or medium for visual review only. '
        f'Allowed hypotheses {sorted(base.HYPOTHESES)}; allowed requested_checks {sorted(base.ACTIONS)}. '
        'First image reference binary mask, second aligned inspection binary mask. Minimal candidates: '
        +json.dumps(minimal,ensure_ascii=False))
    return request,ids

def validate(value,ids):
    if not isinstance(value,dict) or set(value)!={'review_order','regions','summary_zh'}:raise ValueError('Unexpected fields')
    old=copy.deepcopy(value)
    if not isinstance(old['regions'],list):raise ValueError('Invalid regions')
    for row in old['regions']:
        if not isinstance(row,dict):raise ValueError('Invalid region')
        judgment=row.pop('judgment',None);priority=row.pop('suggested_priority',None)
        if not isinstance(judgment,str) or judgment not in JUDGMENTS or not isinstance(priority,str) or priority not in LEVELS:
            raise ValueError('Invalid judgment or priority')
        if judgment=='insufficient_evidence' and priority!='normal':raise ValueError('Unknown evidence must keep priority')
    base.validate(old,ids)
    return copy.deepcopy(value)

def local_evidence(report,ref,ins):
    base.binary_mask(ref);base.binary_mask(ins)
    with Image.open(ref) as im:a=np.asarray(im).astype(bool)
    with Image.open(ins) as im:b=np.asarray(im).astype(bool)
    if a.shape!=b.shape:raise ValueError('Mask dimensions differ')
    alignment=report.get('alignment',{})
    reliable=alignment.get('alignment_quality',{}).get('reliable') is True
    evidence=[]
    for index,candidate in enumerate(report['review_regions'],1):
        x1,y1,x2,y2=base.backend._candidate_xyxy(candidate)
        if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or not 0<=x1<x2<=a.shape[1] or not 0<=y1<y2<=a.shape[0]:raise ValueError('Invalid region frame')
        left,top=math.floor(x1),math.floor(y1);right,bottom=math.ceil(x2),math.ceil(y2)
        difference=int(np.count_nonzero(a[top:bottom,left:right]^b[top:bottom,left:right]))
        strong=candidate.get('tier')=='dino_and_sam'
        evidence.append({'candidate_id':f'candidate_{index:03d}','local_region_index':index,
                         'alignment_reliable':reliable,'mask_changed_pixels':difference,
                         'strong_local_wire_evidence':strong,'baseline_priority':'normal'})
    return evidence

def adjust(result,evidence,current_binding):
    by_id={x['candidate_id']:x for x in evidence}
    usable=result.get('status')=='ok' and result.get('binding')==current_binding
    if usable:
        try:plan=validate(result['plan'],list(by_id))
        except (ValueError,KeyError,TypeError):usable=False
    rows={x['candidate_id']:x for x in plan['regions']} if usable else {}
    audit=[]
    for item in evidence:
        row=rows.get(item['candidate_id']);original=LEVELS[item['baseline_priority']];final=original
        reason='维持本地等级：模型不可用、格式不符或输入已变化。'
        if row:
            wanted=LEVELS[row['suggested_priority']]
            reason='维持本地等级：模型判断或独立证据尚不足以调整。'
            if item['alignment_reliable'] and row['confidence']=='medium':
                if row['judgment']=='visible_change' and wanted>original and item['strong_local_wire_evidence'] and item['mask_changed_pixels']>0:
                    final=min(original+1,wanted,3);reason='升一级：模型判断可见变化，本地配准及线材差异证据支持。'
                elif row['judgment']=='likely_artifact' and wanted<original and not item['strong_local_wire_evidence'] and item['mask_changed_pixels']==0:
                    final=max(original-1,wanted,1);reason='降一级：模型提示外观误差，局部掩膜无变化且无强线材证据；仍需人工复核。'
        audit.append({**item,'llm_judgment':row['judgment'] if row else None,
                      'llm_suggested_priority':row['suggested_priority'] if row else None,
                      'adjusted_priority':next(k for k,v in LEVELS.items() if v==final),
                      'display_label':LABELS[final],'adjustment_reason_zh':reason,
                      'candidate_retained':True,'electrical_verdict':'not_assessed'})
    return {'policy_version':'bounded_visual_review_priority_v1','rows':audit,'model_result_accepted':usable,
            'overall_priority':max((LEVELS[r['adjusted_priority']] for r in audit),default=None),
            'all_candidates_retained':True,'local_decision_unchanged':True}

def run(ref,ins,candidates,settings=None,api_key=None,sender=base.backend._post_json):
    settings=settings or base.backend.load_settings();frozen=copy.deepcopy(candidates)
    result={'status':'fallback_local_review','binding':binding(ref,ins,frozen),'plan':None,
            'notice':base.NOTICE,'requested_model':settings.model,'server_model':None,'network_requests':0,
            'automatic_actions_executed':0,'local_candidates_unchanged':True}
    if not candidates:return {**result,'status':'no_candidates'}
    stage='input'
    result['diagnostic']={'failure_stage':None,'reason':None,'finish_reason':None}
    try:
        if not settings.enabled:raise ValueError('Disabled')
        request,ids=payload(ref,ins,frozen,settings);token=base.backend._api_token(settings,api_key)
        req=Request(settings.endpoint,data=json.dumps(request,ensure_ascii=False).encode(),
                    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST')
        result['network_requests']=1
        stage='transport'
        response=sender(req,settings.timeout_seconds);result['server_model']=response.get('model')
        stage='completion'
        choice=response['choices'][0]
        if choice.get('finish_reason') in ('stop','length','content_filter','tool_calls'):
            result['diagnostic']['finish_reason']=choice['finish_reason']
        parsed=base.backend._json_content(choice['message']['content'])
        stage='validation'
        result['plan']=validate(parsed,ids)
        assert candidates==frozen
        result['status']='ok'
    except (ValueError,TypeError,KeyError,IndexError,OSError,base.backend.DeepSeekMaskReviewError) as error:
        result['safe_error']='风险判断不可用，保留本地复核等级。'
        result['diagnostic'].update(failure_stage=stage,reason=str(error) if str(error) in SAFE_REASONS else 'unavailable_or_invalid_response')
    return result
