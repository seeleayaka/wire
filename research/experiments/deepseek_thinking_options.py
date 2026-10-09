"""Bounded opt-in thinking controls. No network until the review button click."""
import json
from dataclasses import replace
from urllib.request import Request

MODES={'disabled':('关闭思考 · 快速回答',None),
       'low':('轻量思考 · 推荐','low'),
       'high':('标准思考','high'),
       'max':('深度思考 · 较慢','max')}

def configured_settings(settings,mode):
    if mode not in MODES:raise ValueError('Unsupported thinking mode')
    return replace(settings,enabled=True,max_tokens=2400 if mode=='disabled' else 8192,
                   timeout_seconds=45 if mode=='disabled' else 120)

def thinking_sender(mode,transport,photo_first=None,diagnostics=None):
    if mode not in MODES:raise ValueError('Unsupported thinking mode')
    def send(req,timeout):
        body=json.loads(req.data)
        body['thinking']={'type':'disabled' if mode=='disabled' else 'enabled'}
        if mode!='disabled':body['reasoning_effort']=MODES[mode][1]
        else:body.pop('reasoning_effort',None)
        if photo_first:body['messages'][1]['content'][0]['text']+=photo_first
        request=Request(req.full_url,data=json.dumps(body,ensure_ascii=False).encode(),headers=dict(req.header_items()),method='POST')
        response=transport(request,timeout)
        if diagnostics is not None:
            diagnostics['reasoning_present']=any(bool(c.get('message',{}).get('reasoning_content')) for c in response.get('choices',[]))
            diagnostics['finish_reason']=response.get('choices',[{}])[0].get('finish_reason')
        return response
    return send
