"""Two bounded, previously authorized mask-only calls; no mainline mutation."""
import json,re,hashlib,time
from pathlib import Path
from dataclasses import replace
import llm_recheck_planner as p

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/llm_recheck_planner_20261008'
CASES={
    'cabinet2':ROOT/'artifacts/local_performance_fresh_20261007/cabinet_2/desktop_output/20261007_200359/report.json',
    'cabinet4_right1':ROOT/'artifacts/cabinet4_right1_fresh_20261008/cabinet4_right1_changed_view/desktop_output/20261008_190515/report.json',
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    source=Path('C:/Users/HUAWEI/Desktop/中转站信息.txt').read_text(encoding='utf-8')
    urls=list(re.finditer(r'https://api\.deepseek\.com[^\s"\']*',source))
    if len(urls)!=1:raise ValueError('Ambiguous official endpoint')
    keys=re.findall(r'sk-[A-Za-z0-9_-]+',source[max(0,urls[0].start()-300):urls[0].start()])
    if len(keys)!=1:raise ValueError('Ambiguous nearby credential')
    token=keys[0]
    settings=replace(p.backend.load_settings(),endpoint='https://api.deepseek.com/chat/completions',max_tokens=2400,timeout_seconds=45)
    results={}
    for name,path in CASES.items():
        before=sha(path)
        report=json.loads(path.read_text(encoding='utf-8'));fusion=report['sam3_fusion']
        ref=Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
        ins=Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
        start=time.monotonic()
        result=p.run(ref,ins,report['review_regions'],settings,token)
        result.update(case=name,source_report_sha256=before,source_report_unchanged=before==sha(path),
                      elapsed_seconds=round(time.monotonic()-start,3),candidate_count=len(report['review_regions']),
                      raw_images_sent=False,reused_local_visual_outputs=True,sam_rerun=False)
        encoded=json.dumps(result,ensure_ascii=False,indent=2)
        assert token not in encoded
        (OUT/(name+'.json')).write_text(encoded,encoding='utf-8')
        results[name]={'status':result['status'],'candidate_count':result['candidate_count'],
                       'server_model':result['server_model'],'elapsed_seconds':result['elapsed_seconds']}
        print(json.dumps({name:results[name]},ensure_ascii=False),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'results':results,'production_modified':False,'accuracy_gain_measured':False},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
