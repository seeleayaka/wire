"""Compare frozen source controls, not demo accuracy; never alter evidence."""
import json
from pathlib import Path
from run_prompt_contrast import save, digest, verify
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/sam_prompt_confidence_source_20261008'


def assess(rows, prompt, boxed):
    selected = [r for r in rows if r['prompt'] == prompt and r['boxed'] == boxed]
    assert len(selected) == 5 and len({r['id'] for r in selected}) == 5
    reference = next(r for r in selected if r['id'] == 'reference')
    ready = reference['eligible'] == 1
    visible = {r['id'] for r in selected if r['id'] in
               ['reference', 'source_visible_01', 'source_visible_02'] and r['eligible'] == 1}
    conflicts = {r['id'] for r in selected if r['id'].startswith('source_exposed_') and r['high_socket_touch']}
    return ready, visible, conflicts


def main():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    verify(p['pins'])
    assert source_pins() == p['mainline_pins']
    report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete'
    baseline_ready, baseline_visible, baseline_conflicts = assess(report['cases'], 'cable', True)
    old_visible = {'reference', 'source_visible_02'}
    rows = []
    for prompt in p['prompts']:
        for boxed in [False, True]:
            ready, visible, conflicts = assess(report['cases'], prompt, boxed)
            gained = visible - baseline_visible
            lost = (baseline_visible | old_visible) - visible
            added_conflicts = conflicts - baseline_conflicts
            passed = bool(ready and gained and not lost and not conflicts and not added_conflicts)
            rows.append(dict(prompt=prompt, boxed=boxed, reference_ready=ready,
                visible_controls_with_unique_component=sorted(visible),
                new_vs_fresh_baseline=sorted(gained), lost_vs_old_or_fresh=sorted(lost),
                exposed_socket_contradictions=sorted(conflicts), strict_source_gate_passed=passed))
    save(OUT / 'source_gate_report.json', dict(status='complete', rows=rows,
        report_sha256=digest(OUT / 'report.json'), fresh_baseline_reference_ready=baseline_ready,
        baseline_visible=sorted(baseline_visible), baseline_conflicts=sorted(baseline_conflicts),
        no_deployment=True, native_audit_required=True, visual_review_required=True,
        electrical_connections_confirmed=0, not_field_accuracy=True))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
