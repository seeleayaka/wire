"""Two approved mask-only judgments, with all local priorities retained in audit."""
import json
from types import SimpleNamespace
import probe_llm_recheck_planner_20261008 as probe
import llm_review_priority as risk

def main():
    probe.OUT=probe.ROOT/'artifacts/llm_review_priority_20261008'
    probe.p=SimpleNamespace(backend=risk.base.backend,run=risk.run)
    probe.main()
    summary={}
    for name,source in probe.CASES.items():
        report=json.loads(source.read_text(encoding='utf-8'))
        dest=probe.OUT/(name+'.json');result=json.loads(dest.read_text(encoding='utf-8'))
        fusion=report['sam3_fusion']
        ref=risk.Path(fusion['reference_sam3']['output_dir'])/'mask_union.png'
        ins=risk.Path(fusion['inspection_sam3']['output_dir'])/'mask_union.png'
        evidence=risk.local_evidence(report,ref,ins)
        audit=risk.adjust(result,evidence,risk.binding(ref,ins,report['review_regions']))
        result['priority_adjustment']=audit
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        summary[name]=[{'candidate_id':r['candidate_id'],'judgment':r['llm_judgment'],
                       'suggestion':r['llm_suggested_priority'],'final':r['adjusted_priority']} for r in audit['rows']]
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
