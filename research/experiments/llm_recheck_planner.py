"""Experimental mask-only review planner, sharing the existing mainline transport.

Plans are suggestions, never executable tools or replacements for local results.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request
import numpy as np
from PIL import Image

sys.path.insert(0, 'E:/PythonProject10/prototype')
import deepseek_mask_review as backend

ACTIONS = {
    'inspect_raw_region': '回看本地原图与局部掩膜',
    'check_alignment': '核查参考与待检图的配准',
    'inspect_occlusion': '核查标签或线束遮挡',
    'verify_terminal_identity': '由操作员补充可见端点身份',
    'confirm_same_wire': '由操作员确认两段是否属于同一线材',
    'recapture_same_view': '补拍接近参考视角的照片',
    'onsite_continuity_check': '由具备资质人员按现场安全规程补充导通证据',
}
HYPOTHESES = {'appearance_change', 'alignment_artifact', 'occlusion', 'insufficient_evidence'}
NOTICE = '大模型输出为复核建议；保留全部本地候选，电气连接结论由现场证据与人工确认形成。'

def binary_mask(path):
    with Image.open(path) as im:
        if im.mode not in ('1', 'L'): raise ValueError('Only binary masks are permitted')
        values = np.unique(np.asarray(im))
        if not set(values.tolist()).issubset({False, True, 0, 255}):
            raise ValueError('Only binary masks are permitted')

def payload(ref, ins, candidates, settings):
    binary_mask(ref); binary_mask(ins)
    request, ids = backend.build_request_payload(ref, ins, candidates, settings)
    schema = {
        'review_order': ids,
        'regions': [{'candidate_id': ids[0], 'observation_zh': '仅描述可见掩膜差异或证据不足',
                     'hypotheses': ['insufficient_evidence'], 'requested_checks': ['inspect_raw_region'],
                     'question_zh': '请操作员核查什么？', 'confidence': 'low'}],
        'summary_zh': '简短中文复核方案',
    }
    instruction = (
        'Create a human recheck plan from ONLY the two binary masks and the supplied candidate boxes. '
        'Return exactly the JSON schema below. Include every candidate exactly once in regions and review_order. '
        'observation_zh must describe visible mask geometry or explicitly say insufficient evidence, not a confirmed fault. '
        'hypotheses and requested_checks use ONLY the allowed enums. Questions are for an operator, not answers. '
        'Do not infer terminals, cable identity, electrical continuity or correct/wrong wiring; no final verdict. '
        'Do not create, remove, move or merge candidates, no automatic tool execution. '
        'Confidence is low or medium for proposed visual review ONLY, not electrical correctness. '
        f'Allowed hypotheses: {sorted(HYPOTHESES)}. Allowed checks: {sorted(ACTIONS)}. '
        f'Schema: {json.dumps(schema,ensure_ascii=False)}. '
    )
    request['messages'][0]['content'] = 'Return a conservative, structured human recheck plan. JSON only. No electrical verdict.'
    request['messages'][1]['content'][0]['text'] = instruction + request['messages'][1]['content'][0]['text'].split('Local candidates: ')[-1]
    return request, ids

def validate(value, ids):
    if not isinstance(value, dict) or set(value) != {'review_order', 'regions', 'summary_zh'}:
        raise ValueError('Unexpected plan fields')
    order = value['review_order']
    if not isinstance(order,list) or len(order)!=len(ids) or not all(isinstance(x,str) for x in order) or set(order)!=set(ids):
        raise ValueError('Missing, duplicate or invented candidate IDs')
    regions = value['regions']
    if not isinstance(regions,list) or len(regions)!=len(ids): raise ValueError('Incomplete region plan')
    seen = []
    for row in regions:
        if not isinstance(row,dict) or set(row)!={'candidate_id','observation_zh','hypotheses','requested_checks','question_zh','confidence'}:
            raise ValueError('Unexpected region fields')
        if not isinstance(row['candidate_id'],str) or row['candidate_id'] not in ids: raise ValueError('Invented region')
        seen.append(row['candidate_id'])
        for name, allowed in [('hypotheses',HYPOTHESES),('requested_checks',set(ACTIONS))]:
            items=row[name]
            if not isinstance(items,list) or not 1<=len(items)<=len(allowed) or not all(isinstance(x,str) for x in items) or len(set(items))!=len(items) or not set(items)<=allowed:
                raise ValueError('Unknown or duplicate review action/hypothesis')
        if row['confidence'] not in ('low','medium'): raise ValueError('Unsupported confidence')
        for key in ('observation_zh','question_zh'):
            if not isinstance(row[key],str) or not row[key].strip() or len(row[key])>300: raise ValueError('Invalid review text')
    if len(set(seen))!=len(ids): raise ValueError('Duplicate region plan')
    if not isinstance(value['summary_zh'],str) or not value['summary_zh'].strip() or len(value['summary_zh'])>500: raise ValueError('Invalid summary')
    return copy.deepcopy(value)

def run(ref, ins, candidates, settings=None, api_key=None, sender=backend._post_json):
    settings=settings or backend.load_settings()
    frozen=copy.deepcopy(candidates)
    binding={'reference_mask_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),
             'inspection_mask_sha256':hashlib.sha256(ins.read_bytes()).hexdigest(),
             'candidate_sha256':hashlib.sha256(json.dumps(frozen,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    base={'schema_version':1,'notice':NOTICE,'local_candidates_unchanged':True,'automatic_actions_executed':0,
          'binding':binding,'requested_model':settings.model,'server_model':None,'network_requests':0}
    if not candidates:
        return {**base,'status':'no_candidates','plan':None}
    try:
        if not settings.enabled: raise ValueError('Review disabled')
        request,ids=payload(ref,ins,frozen,settings)
        token=backend._api_token(settings,api_key)
        req=Request(settings.endpoint,data=json.dumps(request,ensure_ascii=False).encode(),
                    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST')
        base['network_requests']=1
        response=sender(req,settings.timeout_seconds)
        base['server_model']=response.get('model')
        plan=validate(backend._json_content(response['choices'][0]['message']['content']),ids)
        assert frozen==candidates
        return {**base,'status':'ok','plan':plan}
    except (ValueError,TypeError,KeyError,IndexError,backend.DeepSeekMaskReviewError,OSError):
        # Never expose service bodies, exception text or credentials in a saved report.
        return {**base,'status':'fallback_local_review','plan':None,'safe_error':'规划不可用或格式校验未通过，保留原本地复核流程。'}
