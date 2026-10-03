import json
import sys
from pathlib import Path
from local_review_contract_v2 import validate_candidate_review
from expected_plan_contract import validate_draft
root=Path(__file__).resolve().parents[1]
edition=sys.argv[1] if len(sys.argv)>1 else 'wiremind_workbench_20261002'
bundle=json.loads((root/'artifacts'/edition/'bundle.json').read_text(encoding='utf-8'))
original=json.loads((root/'artifacts/local_review_ui_v4_20261002/bundle.json').read_text(encoding='utf-8'))
assert bundle==original
out=root/'output/playwright'/edition
validate_candidate_review(bundle,json.loads((out/'review.json').read_text(encoding='utf-8')))
validate_draft(bundle,json.loads((out/'plan.json').read_text(encoding='utf-8')))
print('Actual browser exports validated by unchanged Python contracts; source bundle unchanged.')
