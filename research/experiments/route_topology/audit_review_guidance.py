"""Read stored companion records and source pins; no inference or annotations."""
import json
from pathlib import Path
from core import sha256, image_binding
from run_audit import source_pins

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT/'artifacts/review_guidance_ready_v3_20261007'
    catalog = json.loads((folder/'catalog.json').read_text(encoding='utf-8'))
    protocol = json.loads((ROOT/'artifacts/mendeley_confirmed_bundle_demo_20261006/protocol.json').read_text(encoding='utf-8'))
    assert source_pins() == protocol['mainline_pins'], 'E source/dirty tree drift'
    for path, digest in catalog['source_pins'].items():
        assert sha256(path) == digest, 'source drift'
    for case in catalog['cases']:
        assert image_binding(case['source_path']) == case['image_binding']
        assert image_binding(folder/('images/'+case['id']+'.jpg')) == case['image_binding']
    records = []
    for path in sorted(folder.glob('review_*/report.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        p = json.loads(path.with_name('guidance.json').read_text(encoding='utf-8'))
        case = next(c for c in catalog['cases'] if c['id'] == r['case_id'])
        assert r['automatic_comparison_unchanged'] == case['automatic_comparison'] == p['automatic_comparison_unchanged']
        assert r['image_binding'] == case['image_binding'] == p['image_binding']
        assert p['review_report_sha256'] == sha256(path)
        assert p['review_source'] == r['evidence_source'] == 'software_fixture'
        assert p['tools_executed'] == [] and p['model_calls'] == p['network_calls'] == p['automatic_new_hits'] == 0
        assert r['automatic_new_hits'] == 0 and p['electrical_continuity'] == 'not_assessed'
        assert all(a['execution'] == 'not_executed' for a in p['actions'])
        assert 'review_plan' not in r
        records.append({'report': str(path), 'sha256': sha256(path), 'guidance_sha256': sha256(path.with_name('guidance.json'))})
    assert records, 'no persisted records'
    checks = []
    for relative in ['output/playwright/review_guidance_20261007_verified/report.json',
                     'output/playwright/human_recheck_fixes_20261007_guidance_companion_verified/report.json',
                     'output/playwright/human_bundle_recheck_20261007_guidance_companion_verified/report.json']:
        report = json.loads((ROOT/relative).read_text(encoding='utf-8'))
        assert report['status'] == 'PASS'
        checks.append({'report': relative, 'checks': len(report['checks']), 'sha256': sha256(ROOT/relative)})
    output = ROOT/'artifacts/review_guidance_acceptance_20261007'
    output.mkdir(exist_ok=False)
    result = {'status': 'PASS', 'kind': 'software_storage_and_pin_audit_not_vision_accuracy',
              'E_mainline_and_dirty_tree_unchanged': True, 'records': records, 'browser_checks': checks,
              'fresh_inference': False, 'automatic_new_hits': 0, 'human_acceptance': False,
              'deployed_to_E': False, 'rule_guidance_not_LLM': True}
    (output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS', len(records), 'software records;', sum(c['checks'] for c in checks), 'browser checks; E unchanged')


if __name__ == '__main__': main()
