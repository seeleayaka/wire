"""Import a local review into an existing visual Agent task; no automatic verdict."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_agent.local_evidence_bridge import attach_review_file
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--evidence-files',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--plan',type=Path);p.add_argument('--allow-test-records',action='store_true')
    a=p.parse_args();manifest=json.loads(a.evidence_files.read_text(encoding='utf-8'))
    files={case:{key:(a.evidence_files.parent/Path(value)).resolve() for key,value in paths.items()} for case,paths in manifest.items()}
    result=attach_review_file(a.task,a.bundle,a.review,files,a.case,plan_path=a.plan,allow_test_records=a.allow_test_records)
    print(json.dumps({k:result[k] for k in ('attachment_id','case','review_mode','candidate_summary','topology_precheck','connection_edges','automatic_fault_verdict')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
