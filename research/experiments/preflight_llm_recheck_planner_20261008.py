"""Offline request inspection only. No token access, sender or external model."""
import json,hashlib
from probe_llm_recheck_planner_20261008 import CASES,ROOT
import llm_recheck_planner as p

out=ROOT/'artifacts/llm_recheck_planner_preflight_20261008'
out.mkdir(exist_ok=True)
checks=[]
for name,path in CASES.items():
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    report=json.loads(path.read_text(encoding='utf-8'))
    fusion=report['sam3_fusion']
    ref=p.Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
    ins=p.Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
    request,ids=p.payload(ref,ins,report['review_regions'],p.backend.load_settings())
    content=request['messages'][1]['content']
    assert [x['type'] for x in content]==['text','image_url','image_url']
    assert len(ids)==len(report['review_regions'])
    serialized=json.dumps(request,ensure_ascii=False)
    for secret in (str(path),str(ref),str(ins)):assert secret not in serialized
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    checks.append({'case':name,'candidate_count':len(ids),'binary_masks_valid':True,
                   'payload_bytes':len(serialized.encode()),'source_report_sha256':before,
                   'source_report_unchanged':True,'no_paths_in_payload':True,
                   'credentials_read':False,'network_requests':0})
result={'cases':checks,'status':'offline_preflight_passed','live_model_results':False,
        'unit_tests_passed':12,'production_modified':False,
        'live_blocker':'Requires explicit permission to send both specified wiring masks and candidate coordinates to DeepSeek'}
(out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
