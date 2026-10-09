"""One explicitly requested diagnostic replay, same approved masks and prompt.

Does not change production, schemas, priority rules or prior evidence.
"""
import hashlib,json,re,time
from pathlib import Path
from dataclasses import replace
import llm_review_priority as risk
from probe_llm_recheck_planner_20261008 import ROOT,CASES

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out=ROOT/'artifacts/llm_priority_cabinet2_diagnostic_20261008'
    out.mkdir(exist_ok=False)
    source=CASES['cabinet2'];before=sha(source)
    report=json.loads(source.read_text(encoding='utf-8'));fusion=report['sam3_fusion']
    ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
    ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
    settings=replace(risk.base.backend.load_settings(),endpoint='https://api.deepseek.com/chat/completions',max_tokens=2400,timeout_seconds=45)
    captured={}
    def sender(request,timeout):
        response=risk.base.backend._post_json(request,timeout)
        captured.update(response)
        return response
    credentials=Path('C:/Users/HUAWEI/Desktop/中转站信息.txt').read_text(encoding='utf-8')
    urls=list(re.finditer(r'https://api\.deepseek\.com[^\s"\']*',credentials))
    if len(urls)!=1:raise ValueError('Ambiguous official endpoint')
    keys=re.findall(r'sk-[A-Za-z0-9_-]+',credentials[max(0,urls[0].start()-300):urls[0].start()])
    if len(keys)!=1:raise ValueError('Ambiguous adjacent credential')
    token=keys[0]
    start=time.monotonic()
    result=risk.run(ref,ins,report['review_regions'],settings,token,sender)
    detail={'status':result['status'],'elapsed_seconds':round(time.monotonic()-start,3),
            'server_model':result['server_model'],'network_requests':result['network_requests'],
            'source_report_unchanged':before==sha(source),'source_report_sha256':before,
            'risk_code_sha256':sha(Path(risk.__file__)),'binding':result['binding'],
            'raw_images_sent':False,'production_modified':False,'original_response_available':False,
            'validation_error':None,'finish_reason':None}
    parsed=None
    choices=captured.get('choices')
    if isinstance(choices,list) and choices and isinstance(choices[0],dict):
        choice=choices[0]
        finish=choice.get('finish_reason')
        if finish in ('stop','length','content_filter','tool_calls',None):detail['finish_reason']=finish
        content=choice.get('message',{}).get('content')
        detail['content_characters']=len(content) if isinstance(content,str) else None
        try:
            parsed=risk.base.backend._json_content(content)
            risk.validate(parsed,[f'candidate_{i:03d}' for i in range(1,len(report['review_regions'])+1)])
            detail['validation_replay']='passed'
        except (ValueError,TypeError,KeyError,IndexError,risk.base.backend.DeepSeekMaskReviewError) as error:
            # Validator messages are local constant strings, never service bodies.
            detail['validation_error']=str(error)
            detail['validation_replay']='rejected'
        if parsed is not None:
            sanitized=json.dumps(parsed,ensure_ascii=False,indent=2)
            sanitized=sanitized.replace(token,'[REDACTED]')
            sanitized=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',sanitized)
            (out/'response_fixture.json').write_text(sanitized,encoding='utf-8')
    assert detail['source_report_unchanged']
    encoded=json.dumps(detail,ensure_ascii=False,indent=2)
    assert token not in encoded
    (out/'diagnostic.json').write_text(encoded,encoding='utf-8')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(detail,ensure_ascii=False))

if __name__=='__main__':main()
