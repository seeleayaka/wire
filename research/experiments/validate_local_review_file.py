"""Verify downloaded reviews against live evidence versions, no graph mutation."""
import argparse
import hashlib
import json
from pathlib import Path
from inspection_agent.terminal_mapping import image_binding
from local_review_contract import validate_local_review
from local_review_contract_v2 import validate_candidate_review

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('review',type=Path);args=parser.parse_args()
bundle=json.loads((ROOT/'artifacts/local_review_ui_20261001/bundle.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for case,version in bundle['source_versions'].items():
    actual=image_binding(Path(version['image_binding']['image_path']))
    if any(actual[k]!=version['image_binding'][k] for k in ('image_sha256','image_size','coordinate_frame')):
        raise ValueError('live source image changed')
    for key,relative in [('map_sha256',f'artifacts/source_entry_drafts_20261001/{case}_entry_draft.json'),
                         ('endpoint_report_sha256',f'artifacts/sam_crop_coverage_20261001/{case}_source_endpoints.json'),
                         ('audit_sha256',f'artifacts/directional_entry_path_20261001/{case}_audit.json')]:
        if sha(ROOT/relative)!=version[key]:raise ValueError('live evidence version changed')
review=json.loads(args.review.read_text(encoding='utf-8'))
validator=validate_candidate_review if review.get('schema_version')==2 else validate_local_review
print(json.dumps(validator(bundle,review),ensure_ascii=False,indent=2))
