"""Completed fresh masks through real optional backend, without forged ports."""
import json
from pathlib import Path
import sys
from analyze_prompt_contrast import OUT
from run_paired_evidence import load_run
from run_review import save, execute
from build_mask_workbench import execute as workbench


def side(run):
    origin,records=load_run(run)
    return {'image_path':origin['image_path'],'image_binding':origin['image_binding'],
            'ports':[],'mask_ids':list(range(1,len(records)+1)),
            'selection_review':{'confirmed':False},'coverage_review':{'confirmed':False},
            'scope_description':'All original instances retained; no authenticated terminal ports or netlist. This is not a reviewed connection scope.'}


def main():
    report=json.loads((OUT/'inference_report.json').read_text(encoding='utf-8'))
    if report['status']!='complete' or len(report['cases_completed'])!=3:
        raise ValueError('requires completed frozen new inference')
    out=OUT/'actual_pair_backend'
    out.mkdir(exist_ok=False)
    summaries=[]
    for prompt in ['cable','wire']:
        ref=OUT/'crop_cabinet_1'/prompt
        ins=OUT/'crop_cabinet_2'/prompt
        pair={'schema_version':1,'scene_type':'real_cabinet_pair_unconfirmed_draft',
              'reference':side(ref),'inspection':side(ins),'expected_connections':None,
              'expected_review':{'confirmed':False}}
        path=out/f'{prompt}_paired_draft.json'
        save(path,pair)
        result=execute(path,ref,ins,out/f'{prompt}_review')
        assert result['decision']=='insufficient_evidence' and not result['automatic_fault_verdict']
        summaries.append({'prompt':prompt,'decision':result['decision'],'reasons':result['reasons'],
                          'confirmed_connections':0,'model_observer_count':1,
                          'original_masks_reinferred_this_experiment':True,
                          'review_itself_reuses_these_new_bound_masks':True})
        if prompt=='wire':
            workbench(ref,ins,out/'fresh_wire_workbench')
    save(out/'report.json',{'status':'complete','cases':summaries,'ports_fabricated':False,
         'expected_edges_fabricated':False,'old_cues_suppressed':False,'E_deployed':False})
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
