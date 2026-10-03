"""Replay archived source evidence and actual downloaded UI test records into real Agent tasks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from inspection_agent import InspectionTask
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/local_agent_bridge_20261002'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bp=ROOT/'artifacts/local_review_ui_v4_20261002/bundle.json';bundle=load(bp)
    rp=ROOT/'output/playwright/local_review_v4_20261002/automation_review_v4.json'
    pp=ROOT/'output/playwright/local_review_v4_20261002/automation_plan_v2.json'
    manifest={case:dict(map=str(ROOT/f'artifacts/source_entry_drafts_20261001/{case}_entry_draft.json'),
        endpoints=str(ROOT/f'artifacts/sam_crop_coverage_20261001/{case}_source_endpoints.json'),
        audit=str(ROOT/f'artifacts/directional_entry_path_20261001/{case}_audit.json')) for case in bundle['source_versions']}
    mp=OUT/'evidence_files.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    checks=[]
    for case,version in bundle['source_versions'].items():
        image=version['image_binding']['image_path'];task=InspectionTask('TEST_ONLY_'+case,'cabinet_reference_photo',image,image)
        items=[i for i in bundle['items'] if i['case']==case]
        task.record_visual_analysis(dict(decision='local_segment_evidence_manual_review',review_regions=[dict(id=i['item_id']) for i in items]),
            tool_name='archived_sam_source_evidence_replay',source_report=manifest[case]['audit'])
        report=task.to_report();report['replay_context']=dict(mode='automation_fixture',new_visual_inference=False,
            before_after_comparison=False,note='Same source frame for both inputs; archived evidence attachment test, not inspection accuracy.')
        task=InspectionTask.from_report(report);tp=OUT/(case+'_task.json');task.save(tp)
        base=[sys.executable,'E:/PythonProject10/tools/import_local_evidence_review.py','--task',str(tp),'--bundle',str(bp),
              '--review',str(rp),'--evidence-files',str(mp),'--case',case,'--plan',str(pp)]
        before=tp.read_bytes();rejected=subprocess.run(base,capture_output=True,text=True,encoding='utf-8')
        assert rejected.returncode!=0 and tp.read_bytes()==before
        accepted=subprocess.run(base+['--allow-test-records'],capture_output=True,text=True,encoding='utf-8')
        if accepted.returncode:raise RuntimeError(accepted.stderr)
        reloaded=InspectionTask.load(tp).to_report();local=reloaded['local_evidence_reviews'][0]
        assert reloaded['state']=='awaiting_human_review' and not reloaded['human_conclusions'] and not reloaded['topology_assessments']
        assert not local['connection_edges'] and local['expected_plan_draft']['confirmed'] is False
        before=tp.read_bytes();duplicate=subprocess.run(base+['--allow-test-records'],capture_output=True,text=True,encoding='utf-8')
        assert duplicate.returncode!=0 and tp.read_bytes()==before
        checks.append(dict(case=case,default_fixture_rejected=True,imported=True,duplicate_rejected=True,
            candidate_summary=local['candidate_summary'],state=reloaded['state'],task_sha256=hashlib.sha256(tp.read_bytes()).hexdigest()))
    (OUT/'verification.json').write_text(json.dumps(dict(passed=True,cases=checks,new_visual_inference=False,
        human_confirmation=False,topology_comparison_performed=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
