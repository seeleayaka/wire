"""Validate draft contract and current source versions; no confirmation writes."""
import argparse
import hashlib
import json
from pathlib import Path
from inspection_agent.terminal_mapping import image_binding
from expected_plan_contract import validate_draft
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('draft',type=Path);args=parser.parse_args()
    bundle=json.loads((ROOT/'artifacts/local_review_ui_v4_20261002/bundle.json').read_text(encoding='utf-8'))
    result=validate_draft(bundle,json.loads(args.draft.read_text(encoding='utf-8')))
    for case,version in bundle['source_versions'].items():
        actual=image_binding(Path(version['image_binding']['image_path']))
        if any(actual[k]!=version['image_binding'][k] for k in ('image_sha256','image_size','coordinate_frame')):raise ValueError('live image changed')
        for key,rel in [('map_sha256',f'artifacts/source_entry_drafts_20261001/{case}_entry_draft.json'),('endpoint_report_sha256',f'artifacts/sam_crop_coverage_20261001/{case}_source_endpoints.json'),('audit_sha256',f'artifacts/directional_entry_path_20261001/{case}_audit.json')]:
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=version[key]:raise ValueError('live evidence changed')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
